"""Take the best config from the multi-head sweep and re-train for longer.

Useful to confirm the model has converged + push val loss lower than what
the original sweep achieved.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from data_utils import (
    LOSSES_DIR, SEED, build_vocab, encode, load_text, pick_device, save_run,
    train_test_split,
)
from _nn_common import TrainConfig, count_params, train

# Reuse the model classes from script 05
import importlib.util
spec = importlib.util.spec_from_file_location(
    "tx5", "/Users/pmigdal/my_repos/falenty-gpt-2026/scripts/05_transformer_multi_head.py")
tx5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tx5)

MiniGPT = tx5.MiniGPT
generate_text = tx5.generate_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--embd", type=int, default=128)
    ap.add_argument("--head", type=int, default=4)
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tag", type=str, default="06_extended")
    args = ap.parse_args()

    device = pick_device()
    text = load_text()
    vocab = build_vocab(text)
    data = encode(text, vocab)
    train_data, test_data = train_test_split(data)

    torch.manual_seed(SEED)
    model = MiniGPT(vocab.size, args.embd, args.head, args.layer, args.block, args.dropout).to(device)
    cfg = TrainConfig(block_size=args.block, batch_size=args.batch, lr=args.lr,
                      max_iters=args.iters, eval_every=max(args.iters // 30, 100),
                      eval_iters=20)
    n_params = count_params(model)
    print(f"=== {args.tag} | block={args.block} embd={args.embd} head={args.head} "
          f"layer={args.layer} dropout={args.dropout} | params={n_params:,} ===")

    t0 = time.time()
    iters, tr, vl = train(model, train_data, test_data, cfg, device, verbose=True)
    train_time = time.time() - t0
    print(f"Trained in {train_time:.1f}s; best_val={min(vl):.4f}")

    samples = {
        "Litwo_T1.0": generate_text(model, vocab, "Litwo, ojczyzno moja",
                                     length=600, temperature=1.0, device=device),
        "Litwo_T0.7": generate_text(model, vocab, "Litwo, ojczyzno moja",
                                     length=600, temperature=0.7, device=device),
        "Soplica_T0.8": generate_text(model, vocab, "Soplica",
                                       length=400, temperature=0.8, device=device),
        "Telimena_T0.8": generate_text(model, vocab, "Telimena",
                                       length=400, temperature=0.8, device=device),
    }
    save_run(name=args.tag,
             hyperparams={"block_size": args.block, "n_embd": args.embd, "n_head": args.head,
                          "n_layer": args.layer, "dropout": args.dropout,
                          "lr": args.lr, "batch_size": args.batch, "max_iters": args.iters,
                          "n_params": n_params},
             train_losses=tr, val_losses=vl, iters=iters, samples=samples,
             extra={"train_time_s": train_time, "best_val": min(vl)})


if __name__ == "__main__":
    main()
