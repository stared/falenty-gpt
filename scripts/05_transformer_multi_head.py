"""Mini-GPT: full multi-head causal transformer.

Multi-head self-attention + feed-forward + residual + LayerNorm,
stacked n_layer times. Same data and split as other scripts.
Sweep over block_size, n_embd, n_head, n_layer.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import (
    LOSSES_DIR, SEED, build_vocab, encode, load_text, pick_device, save_run,
    train_test_split,
)
from _nn_common import TrainConfig, count_params, train


class Head(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x); q = self.query(x); v = self.value(x)
        w = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        return w @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int,
                 block_size: int, dropout: float):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


@torch.no_grad()
def generate_text(model: MiniGPT, vocab, start: str, length: int, temperature: float, device: str) -> str:
    model.eval()
    if not start:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        idx = torch.tensor([[vocab.char2id[c] for c in start]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=length, temperature=temperature)
    return "".join(vocab.id2char[i.item()] for i in out[0])


def run_one(block_size: int, n_embd: int, n_head: int, n_layer: int, dropout: float,
            max_iters: int, lr: float, batch_size: int, device: str,
            train_data: torch.Tensor, test_data: torch.Tensor, vocab,
            tag: str) -> dict:
    torch.manual_seed(SEED)
    model = MiniGPT(vocab.size, n_embd, n_head, n_layer, block_size, dropout).to(device)
    cfg = TrainConfig(block_size=block_size, batch_size=batch_size, lr=lr,
                      max_iters=max_iters, eval_every=max(max_iters // 25, 50),
                      eval_iters=20)
    n_params = count_params(model)
    print(f"\n=== {tag} | block={block_size} embd={n_embd} head={n_head} layer={n_layer} | "
          f"params={n_params:,} ===")
    t0 = time.time()
    iters, tr, vl = train(model, train_data, test_data, cfg, device,
                          weight_decay=0.0, verbose=False)
    train_time = time.time() - t0
    print(f"  trained in {train_time:.1f}s; final train={tr[-1]:.4f}  val={vl[-1]:.4f}  "
          f"best_val={min(vl):.4f}")

    samples = {
        "Litwo_T1.0": generate_text(model, vocab, "Litwo, ojczyzno moja",
                                     length=500, temperature=1.0, device=device),
        "Litwo_T0.7": generate_text(model, vocab, "Litwo, ojczyzno moja",
                                     length=500, temperature=0.7, device=device),
        "Soplica_T0.8": generate_text(model, vocab, "Soplica",
                                       length=400, temperature=0.8, device=device),
    }
    save_run(name=tag,
             hyperparams={"block_size": block_size, "n_embd": n_embd, "n_head": n_head,
                          "n_layer": n_layer, "dropout": dropout, "lr": lr,
                          "batch_size": batch_size, "max_iters": max_iters,
                          "n_params": n_params},
             train_losses=tr, val_losses=vl, iters=iters, samples=samples,
             extra={"train_time_s": train_time, "best_val": min(vl)})
    return {"block_size": block_size, "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
            "n_params": n_params, "final_train": tr[-1], "final_val": vl[-1],
            "best_val": min(vl), "time_s": train_time}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="single quick run for smoke test")
    ap.add_argument("--full", action="store_true",
                    help="include heavy configs (>1M params, b=128); ~2h on CPU")
    args = ap.parse_args()
    device = pick_device()
    print(f"Device: {device}")

    text = load_text()
    vocab = build_vocab(text)
    data = encode(text, vocab)
    train_data, test_data = train_test_split(data)

    results: list[dict] = []
    if args.quick:
        results.append(run_one(block_size=32, n_embd=64, n_head=4, n_layer=2, dropout=0.0,
                               max_iters=1000, lr=3e-3, batch_size=32,
                               device=device, train_data=train_data,
                               test_data=test_data, vocab=vocab, tag="05_minigpt_quick"))
    else:
        # Default sweep — finishable in ~30 min on CPU, faster on MPS/CUDA.
        # Heaviest configs (>1M params) only run with --full.
        sweep = [
            # (block, embd, head, layer, dropout, iters)
            (32, 64, 4, 2, 0.0, 3000),    # ~75s on CPU, val~1.93
            (32, 64, 4, 4, 0.0, 3000),    # ~150s, val~2.15 (undertrained)
            (64, 64, 4, 2, 0.0, 3000),    # ~120s, val~1.88 (sweet spot at this size)
            (64, 64, 4, 4, 0.0, 3000),    # ~250s, val~2.16
            (64, 96, 4, 4, 0.0, 3000),    # ~360s, val~2.02
            (64, 96, 4, 6, 0.1, 4000),    # ~900s, val~1.92
        ]
        if args.full:
            sweep += [
                # Heavy configs (b=128) — only with --full
                (128, 96, 4, 4, 0.1, 4000),    # ~870s
                (128, 128, 4, 4, 0.1, 4000),   # ~890s
                (128, 128, 4, 6, 0.1, 6000),   # ~1800s, val~1.81
                (128, 192, 6, 6, 0.1, 6000),   # ~2700s, val~1.73 (best in sweep)
            ]
        for bs, ne, nh, nl, dp, it in sweep:
            tag = f"05_minigpt_b{bs}_e{ne}_h{nh}_l{nl}"
            # AdamW LR schedule note: use 3e-4 for bigger models, 3e-3 for tiny
            lr = 3e-4 if (ne >= 96 or nl >= 4) else 3e-3
            batch = 64 if bs <= 64 else 32
            results.append(run_one(block_size=bs, n_embd=ne, n_head=nh, n_layer=nl,
                                   dropout=dp, max_iters=it, lr=lr, batch_size=batch,
                                   device=device, train_data=train_data,
                                   test_data=test_data, vocab=vocab, tag=tag))

    print("\nSweep summary:")
    print(f"{'block':>5} {'embd':>5} {'head':>4} {'layer':>5} {'params':>10}  "
          f"{'train':>8}  {'val':>8}  {'best_val':>9}  time")
    for r in results:
        print(f"{r['block_size']:>5} {r['n_embd']:>5} {r['n_head']:>4} {r['n_layer']:>5} "
              f"{r['n_params']:>10,}  {r['final_train']:>8.4f}  {r['final_val']:>8.4f}  "
              f"{r['best_val']:>9.4f}  {r['time_s']:>6.1f}s")

    import json
    with open(LOSSES_DIR / "05_multihead_sweep.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
