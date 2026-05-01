"""Single-head transformer (causal self-attention).

Predicts next char at every position simultaneously (sequence model).
Sweep over block_size (context length) and n_embd / head_size.
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
    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x); q = self.query(x); v = self.value(x)
        w = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        return w @ v


class SingleHeadTransformer(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.head = Head(n_embd, head_size, block_size)
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.head(x)
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
def generate_text(model: SingleHeadTransformer, vocab, start: str, length: int,
                  temperature: float, device: str) -> str:
    model.eval()
    if not start:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        idx = torch.tensor([[vocab.char2id[c] for c in start]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=length, temperature=temperature)
    return "".join(vocab.id2char[i.item()] for i in out[0])


def run_one(block_size: int, n_embd: int, head_size: int,
            max_iters: int, lr: float, batch_size: int, device: str,
            train_data: torch.Tensor, test_data: torch.Tensor, vocab,
            tag: str) -> dict:
    torch.manual_seed(SEED)
    model = SingleHeadTransformer(vocab.size, n_embd, head_size, block_size).to(device)
    cfg = TrainConfig(block_size=block_size, batch_size=batch_size, lr=lr,
                      max_iters=max_iters, eval_every=max(max_iters // 25, 50),
                      eval_iters=20)
    n_params = count_params(model)
    print(f"\n=== {tag} | block={block_size} n_embd={n_embd} head={head_size} | "
          f"params={n_params:,} ===")
    t0 = time.time()
    iters, tr, vl = train(model, train_data, test_data, cfg, device, verbose=False)
    train_time = time.time() - t0
    print(f"  trained in {train_time:.1f}s; final train={tr[-1]:.4f}  val={vl[-1]:.4f}  "
          f"best_val={min(vl):.4f}")

    samples = {
        "Litwo_T1.0": generate_text(model, vocab, "Litwo, ojczyzno moja",
                                     length=400, temperature=1.0, device=device),
        "Litwo_T0.7": generate_text(model, vocab, "Litwo, ojczyzno moja",
                                     length=400, temperature=0.7, device=device),
    }
    save_run(name=tag,
             hyperparams={"block_size": block_size, "n_embd": n_embd, "head_size": head_size,
                          "lr": lr, "batch_size": batch_size, "max_iters": max_iters,
                          "n_params": n_params},
             train_losses=tr, val_losses=vl, iters=iters, samples=samples,
             extra={"train_time_s": train_time, "best_val": min(vl)})
    return {"block_size": block_size, "n_embd": n_embd, "head_size": head_size,
            "n_params": n_params, "final_train": tr[-1], "final_val": vl[-1],
            "best_val": min(vl), "time_s": train_time}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    device = pick_device()
    print(f"Device: {device}")

    text = load_text()
    vocab = build_vocab(text)
    data = encode(text, vocab)
    train_data, test_data = train_test_split(data)

    results: list[dict] = []
    if args.quick:
        results.append(run_one(block_size=16, n_embd=32, head_size=32,
                               max_iters=1500, lr=3e-3, batch_size=64,
                               device=device, train_data=train_data,
                               test_data=test_data, vocab=vocab,
                               tag="04_singlehead_quick"))
    else:
        # Sweep
        sweep = [
            # (block_size, n_embd, head_size)
            (8, 32, 32),
            (16, 32, 32),
            (32, 32, 32),
            (64, 32, 32),
            (128, 32, 32),
            # Vary embedding
            (32, 16, 16),
            (32, 64, 64),
            (32, 96, 96),
            # Decouple n_embd and head_size
            (32, 64, 32),
            (32, 64, 128),
            # Bigger
            (64, 64, 64),
            (64, 96, 96),
        ]
        for bs, ne, hs in sweep:
            tag = f"04_singlehead_b{bs}_e{ne}_h{hs}"
            results.append(run_one(block_size=bs, n_embd=ne, head_size=hs,
                                   max_iters=3000, lr=3e-3, batch_size=64,
                                   device=device, train_data=train_data,
                                   test_data=test_data, vocab=vocab, tag=tag))

    print("\nSweep summary:")
    print(f"{'block':>5} {'embd':>5} {'head':>5} {'params':>9}  {'train':>8}  {'val':>8}  {'best_val':>9}  time")
    for r in results:
        print(f"{r['block_size']:>5} {r['n_embd']:>5} {r['head_size']:>5} "
              f"{r['n_params']:>9,}  {r['final_train']:>8.4f}  {r['final_val']:>8.4f}  "
              f"{r['best_val']:>9.4f}  {r['time_s']:>6.1f}s")

    import json
    with open(LOSSES_DIR / "04_singlehead_sweep.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
