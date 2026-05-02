"""Mini-GPT char-level na korpusie Wolnych Lektur, do odpalenia na GPU.

Single-file, standalone. Konfiguracje na samej górze pliku.

Workflow:
    1. ./for_gpu/prepare.sh             # pobiera + skleja → data/corpus.txt
    2. python for_gpu/train.py PRESET   # PRESET = tiny | batch256 | ctx2048
       (bez argumentu = "tiny")

Wynik: data/checkpoint_<preset>.pt + data/sample_<preset>.txt + data/loss_<preset>.json

Sprzęt — szacowane czasy + mieszczenie się w pamięci (bf16):

    GPU            VRAM    tiny    batch256   ctx2048
    ───────────    ─────   ─────   ────────   ───────
    T4             16 GB   ~15 m   ~75 m      OOM (za mało VRAM)
    RTX 5090       32 GB   ~1 m    ~6 m       ~25 m
    A100 40 GB     40 GB   ~3 m    ~15 m      ciasno (zmniejsz batch=8)
    A100 80 GB     80 GB   ~3 m    ~15 m      ~45 m
    H100 80 GB     80 GB   ~1 m    ~5 m       ~15 m

Uwagi:
- T4 nie ma bf16 → spada do fp32 (~3× wolniej, ryzyko niestabilności).
  Dla T4 zacznij od `tiny`. `ctx2048` na T4 nie ma szans (attention O(B·T²)).
- A100/H100 mają bf16 i są stabilne dla wszystkich preset'ów.
- RTX 5090 (Blackwell) ma fp8, ale skrypt go nie używa — bf16 wystarczająco szybkie.
- Czasy szacunkowe; zależą od sterownika i wersji PyTorcha.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# Znak zastępczy dla rzadkich/obcych liter (Greckie cytaty, sanskryt, cyrylica
# w cytatach itp). Unicode REPLACEMENT CHARACTER — kanoniczny placeholder.
RARE_CHAR = "�"  # widoczny jako: �

# Folder z danymi - domyślnie "data" w cwd, można zmienić env var.
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))


# ====================================================================
# KONFIGURACJE — wybierz przez `python for_gpu/train.py PRESET`
# Tabela czasów na różnych GPU jest w docstringu na górze pliku.
# ====================================================================

@dataclass
class Config:
    block_size: int       # KONTEKST — ile znaków wstecz model widzi
    batch_size: int       # paczek równolegle (więcej = większe zużycie VRAM)
    n_embd: int           # wymiar embeddingu (musi być podzielny przez n_head)
    n_head: int           # liczba głów uwagi
    n_layer: int          # liczba bloków transformera
    dropout: float        # 0.0 dla małych modeli, 0.1 dla większych
    learning_rate: float
    max_iters: int        # liczba iteracji (paczek) treningu
    eval_every: int       # co ile iteracji ewaluacja na teście
    eval_iters: int = 50  # ile paczek do średniej val loss
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # Znaki występujące rzadziej niż min_char_freq× w korpusie zostają
    # zastąpione RARE_CHAR. Bez tego full-Wolnelektury daje 500+ znaków
    # (cytaty grec/łac/rus, glify typograficzne). 5e-6 daje ~150 znaków.
    min_char_freq: float = 5e-6


PRESETS: dict[str, Config] = {
    # Smoke test — mały model, krótki blok, szybko sprawdzasz że pipeline działa.
    "tiny": Config(
        block_size=256, batch_size=128,
        n_embd=192, n_head=6, n_layer=4,
        dropout=0.0, learning_rate=3e-3,
        max_iters=2000, eval_every=200,
    ),  # ~3 M params

    # Wysoki batch size — H100 lubi duży batch, krótki blok = niska pamięć.
    "batch256": Config(
        block_size=256, batch_size=256,
        n_embd=384, n_head=6, n_layer=6,
        dropout=0.1, learning_rate=1e-3,
        max_iters=8000, eval_every=400,
    ),  # ~13 M params

    # Długi kontekst — cały akapit. Batch mały, bo attention skaluje się O(T²).
    "ctx2048": Config(
        block_size=2048, batch_size=16,
        n_embd=512, n_head=8, n_layer=8,
        dropout=0.1, learning_rate=5e-4,
        max_iters=15000, eval_every=750,
    ),  # ~30 M params (~ GPT-2 small skali)
}


# ====================================================================
# Model — taki sam mini-GPT jak w notebooku 5
# ====================================================================

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        w = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        return w @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.Sequential(*[
            Block(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout) for _ in range(cfg.n_layer)
        ])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ====================================================================
# Trening
# ====================================================================

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def main():
    preset_name = sys.argv[1] if len(sys.argv) > 1 else "5min"
    if preset_name not in PRESETS:
        print(f"Nieznany preset: {preset_name}. Dostępne: {list(PRESETS)}")
        sys.exit(1)
    cfg = PRESETS[preset_name]

    device = pick_device()
    print(f"=== Mini-GPT: preset={preset_name} | device={device} ===")
    print(f"  kontekst (block_size): {cfg.block_size} znaków")
    print(f"  batch_size:            {cfg.batch_size}")
    print(f"  n_embd / n_head / n_layer: {cfg.n_embd} / {cfg.n_head} / {cfg.n_layer}")
    print(f"  dropout:               {cfg.dropout}")
    print(f"  learning_rate:         {cfg.learning_rate}")
    print(f"  max_iters:             {cfg.max_iters}  (eval co {cfg.eval_every})")

    # Dane: szukamy data/corpus.txt względem cwd, fallback do katalogu skryptu.
    candidates = [DATA_DIR, Path(__file__).resolve().parent.parent / "data"]
    data_dir = next((d for d in candidates if (d / "corpus.txt").exists()), None)
    if data_dir is None:
        print(f"Brak corpus.txt w żadnej z lokalizacji: {[str(c) for c in candidates]}")
        print(f"Najpierw uruchom: ./scripts/h100_prepare.sh")
        sys.exit(1)
    corpus_path = data_dir / "corpus.txt"

    print(f"Wczytuję {corpus_path} ...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  korpus: {len(text):,} znaków")

    # Filtrujemy rzadkie znaki (cytaty grec/łac/rus, glify) → RARE_CHAR placeholder.
    # NIE spacja — spacja niesie znaczenie (granica słów); rzadki znak musi być
    # czymś innym, by model nauczył się go ignorować.
    from collections import Counter
    counts = Counter(text)
    threshold = max(2, int(cfg.min_char_freq * len(text)))
    rare = {c for c, n in counts.items() if n < threshold}
    if rare:
        print(f"  filtruję: {len(rare)} rzadkich znaków (próg < {threshold}× w korpusie) → {RARE_CHAR!r}")
        text = "".join(RARE_CHAR if c in rare else c for c in text)

    # RARE_CHAR zawsze w vocab (nawet jeśli nic nie zostało odfiltrowane).
    chars = sorted(set(text) | {RARE_CHAR})
    vocab_size = len(chars)
    char2id = {c: i for i, c in enumerate(chars)}
    id2char = {i: c for i, c in enumerate(chars)}
    print(f"  vocab:  {vocab_size} unikalnych znaków (włącznie z {RARE_CHAR!r})")

    data = torch.tensor([char2id[c] for c in text], dtype=torch.long)
    split = int(0.95 * len(data))
    train_data, test_data = data[:split], data[split:]
    print(f"  split:  train={len(train_data):,}, test={len(test_data):,}")

    # Model
    torch.manual_seed(42)
    model = MiniGPT(vocab_size, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model:  {n_params:,} params (~{n_params/1e6:.1f}M)")

    # AdamW + bf16 autocast (na cuda; na cpu/mps fallback do fp32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # === pętla treningowa ===
    history = {"iters": [], "train_loss": [], "val_loss": []}
    best_val = float("inf")
    t_start = time.time()

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for split_name, src in [("train", train_data), ("val", test_data)]:
            losses = []
            for _ in range(cfg.eval_iters):
                xb, yb = get_batch(src, cfg.block_size, cfg.batch_size, device)
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    logits = model(xb)
                    B, T, V = logits.shape
                    loss = F.cross_entropy(logits.view(B*T, V), yb.view(B*T))
                losses.append(loss.item())
            out[split_name] = sum(losses) / len(losses)
        model.train()
        return out

    print(f"\nStart treningu: {cfg.max_iters} iteracji, eval co {cfg.eval_every}.")
    model.train()
    for it in range(cfg.max_iters):
        xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(xb)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), yb.view(B*T))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if it % cfg.eval_every == 0 or it == cfg.max_iters - 1:
            elapsed = time.time() - t_start
            losses = estimate_loss()
            history["iters"].append(it)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])
            print(f"  it {it:>6d}/{cfg.max_iters}  "
                  f"train={losses['train']:.4f}  val={losses['val']:.4f}  "
                  f"({elapsed:.0f}s)")

            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt_path = data_dir / f"checkpoint_{preset_name}.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "config": cfg.__dict__,
                    "vocab": chars,
                    "char2id": char2id,
                    "iter": it,
                    "val_loss": losses["val"],
                }, ckpt_path)

    train_time = time.time() - t_start
    print(f"\nTrening zakończony w {train_time/60:.1f} min. Najlepszy val={best_val:.4f}")

    # === generowanie ===
    print("\nGeneruję próbki tekstu ...")
    model.eval()
    starts = ["Litwo, ojczyzno moja", "\n\n", "I rzekł"]
    samples = {}
    for start in starts:
        idx = torch.tensor([[char2id.get(c, 0) for c in start]], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=600, temperature=0.8, top_k=40)
        samples[start] = "".join(id2char[i.item()] for i in out[0])
        print(f"\n--- start: {start!r} ---")
        print(samples[start])

    # Zapis wyników
    sample_path = data_dir / f"sample_{preset_name}.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        for start, txt in samples.items():
            f.write(f"=== start: {start!r} ===\n{txt}\n\n")

    loss_path = data_dir / f"loss_{preset_name}.json"
    with open(loss_path, "w", encoding="utf-8") as f:
        json.dump({
            "preset": preset_name,
            "config": cfg.__dict__,
            "n_params": n_params,
            "vocab_size": vocab_size,
            "train_time_s": train_time,
            "best_val": best_val,
            "history": history,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nZapisane:")
    print(f"  checkpoint:  data/checkpoint_{preset_name}.pt")
    print(f"  próbki:      data/sample_{preset_name}.txt")
    print(f"  loss curve:  data/loss_{preset_name}.json")


if __name__ == "__main__":
    main()
