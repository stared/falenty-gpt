"""Linear (logistic regression) model.

Embeds last `context_size` chars, flattens, applies a single linear layer
to vocab logits. Sweep over context_size and embedding_dim.

Predicts only the LAST position from a fixed-size context (unlike the
transformer scripts, which predict every position in parallel).
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import (
    LOSSES_DIR, SEED, build_vocab, encode, load_text, pick_device, save_run,
    train_test_split,
)
from _nn_common import TrainConfig, count_params, train


class LinearModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_size: int):
        super().__init__()
        self.context_size = context_size
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(context_size * embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, context_size)  ->  logits: (B, vocab_size)
        e = self.emb(x)                 # (B, T, D)
        e = e.reshape(e.size(0), -1)    # (B, T*D)
        return self.linear(e)


def make_batch_fixed(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Batch where x is context of size `block_size` and y is single next char."""
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def generate(model: LinearModel, vocab, start: str, length: int, temperature: float,
             device: str) -> str:
    model.eval()
    ctx = list(start)
    if len(ctx) < model.context_size:
        ctx = [" "] * (model.context_size - len(ctx)) + ctx
    ids = [vocab.char2id.get(c, vocab.char2id[" "]) for c in ctx]
    out = list(start)
    for _ in range(length):
        x = torch.tensor([ids[-model.context_size:]], dtype=torch.long, device=device)
        logits = model(x)[0] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        nid = torch.multinomial(probs, num_samples=1).item()
        out.append(vocab.id2char[nid])
        ids.append(nid)
    return "".join(out)


def run_one(context_size: int, embedding_dim: int, max_iters: int,
            lr: float, batch_size: int, device: str,
            train_data: torch.Tensor, test_data: torch.Tensor, vocab,
            tag: str) -> dict:
    torch.manual_seed(SEED)
    model = LinearModel(vocab.size, embedding_dim, context_size).to(device)
    cfg = TrainConfig(block_size=context_size, batch_size=batch_size, lr=lr,
                      max_iters=max_iters, eval_every=max(max_iters // 25, 50),
                      eval_iters=20)

    n_params = count_params(model)
    print(f"\n=== {tag} | ctx={context_size} emb={embedding_dim} | params={n_params:,} ===")
    t0 = time.time()
    iters, tr, vl = train(model, train_data, test_data, cfg, device,
                          make_batch=make_batch_fixed, verbose=False)
    train_time = time.time() - t0
    print(f"  trained in {train_time:.1f}s; final train={tr[-1]:.4f}  val={vl[-1]:.4f}  best_val={min(vl):.4f}")

    samples = {
        f"Litwo_T1.0": generate(model, vocab, "Litwo, ojczyzno moja",
                                length=300, temperature=1.0, device=device),
        f"Litwo_T0.7": generate(model, vocab, "Litwo, ojczyzno moja",
                                length=300, temperature=0.7, device=device),
    }
    save_run(name=tag,
             hyperparams={"context_size": context_size, "embedding_dim": embedding_dim,
                          "lr": lr, "batch_size": batch_size, "max_iters": max_iters,
                          "n_params": n_params},
             train_losses=tr, val_losses=vl, iters=iters,
             samples=samples,
             extra={"train_time_s": train_time, "best_val": min(vl)})
    return {"context_size": context_size, "embedding_dim": embedding_dim,
            "n_params": n_params, "final_train": tr[-1], "final_val": vl[-1],
            "best_val": min(vl), "time_s": train_time}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="single quick run for testing")
    args = ap.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    text = load_text()
    vocab = build_vocab(text)
    data = encode(text, vocab)
    train_data, test_data = train_test_split(data)

    results: list[dict] = []
    if args.quick:
        results.append(run_one(context_size=4, embedding_dim=16, max_iters=2000, lr=3e-3,
                               batch_size=128, device=device,
                               train_data=train_data, test_data=test_data, vocab=vocab,
                               tag="02_linear_quick"))
    else:
        sweep = [
            # (ctx, emb)
            (1, 16), (2, 16), (3, 16), (5, 16), (8, 16), (16, 16),
            (3, 8), (3, 32), (3, 64),
            (8, 32), (8, 64),
        ]
        for ctx, emb in sweep:
            tag = f"02_linear_ctx{ctx}_emb{emb}"
            results.append(run_one(context_size=ctx, embedding_dim=emb,
                                   max_iters=4000, lr=3e-3, batch_size=256,
                                   device=device,
                                   train_data=train_data, test_data=test_data,
                                   vocab=vocab, tag=tag))

    print("\nSweep summary:")
    print(f"{'ctx':>4} {'emb':>4} {'params':>9}  {'train':>8}  {'val':>8}  {'best_val':>9}  time")
    for r in results:
        print(f"{r['context_size']:>4} {r['embedding_dim']:>4} {r['n_params']:>9,}  "
              f"{r['final_train']:>8.4f}  {r['final_val']:>8.4f}  {r['best_val']:>9.4f}  "
              f"{r['time_s']:>6.1f}s")

    import json
    with open(LOSSES_DIR / "02_linear_sweep.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
