"""Mini-GPT z budżetem czasowym ~1, ~5, ~10 minut.

Bierzemy taki rozmiar modelu i taką liczbę iteracji, by trening kończył się
w docelowym czasie (na CPU M1 Pro). Sprawdzamy, jaki najmniejszy val loss
da się dostać przy ograniczonym budżecie obliczeniowym.

Nie szukamy najlepszego val (w głównym sweepie 6h dało val=1.71). Chcemy
pokazać krzywą time vs val.
"""
from __future__ import annotations

import time

import torch

from data_utils import (
    LOSSES_DIR, SEED, build_vocab, encode, load_text, pick_device, save_run,
    train_test_split,
)
from _nn_common import TrainConfig, count_params, train

import importlib.util
spec = importlib.util.spec_from_file_location(
    "tx5", "/Users/pmigdal/my_repos/falenty-gpt-2026/scripts/05_transformer_multi_head.py")
tx5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tx5)

MiniGPT = tx5.MiniGPT
generate_text = tx5.generate_text


# Configs dobierane tak, by mieścić się w budżecie czasowym na CPU M1 Pro.
# Na MPS/CUDA będą znacznie szybsze (3-10x), więc te budżety to "górna granica".
SPEED_CONFIGS = [
    # tag, block, embd, head, layer, dropout, iters, lr, batch
    ("07_1min",  32, 64,  4, 2, 0.0, 2500, 3e-3, 32),   # ~60s na CPU
    ("07_5min",  64, 96,  4, 4, 0.0, 4000, 3e-3, 32),   # ~5 min na CPU
    ("07_10min", 64, 128, 4, 4, 0.0, 5000, 1e-3, 32),   # ~10 min na CPU
]


def run_one(tag, block, embd, head, layer, dropout, iters, lr, batch,
            device, train_data, test_data, vocab) -> dict:
    torch.manual_seed(SEED)
    model = MiniGPT(vocab.size, embd, head, layer, block, dropout).to(device)
    cfg = TrainConfig(block_size=block, batch_size=batch, lr=lr,
                      max_iters=iters, eval_every=max(iters // 20, 50),
                      eval_iters=20)
    n_params = count_params(model)
    print(f"\n=== {tag} | block={block} embd={embd} head={head} layer={layer} | "
          f"params={n_params:,} | iters={iters} ===")

    t0 = time.time()
    iters_log, tr, vl = train(model, train_data, test_data, cfg, device, verbose=False)
    train_time = time.time() - t0
    best_val = min(vl)
    print(f"  trained in {train_time:.1f}s ({train_time/60:.1f} min); "
          f"final train={tr[-1]:.4f}  val={vl[-1]:.4f}  best_val={best_val:.4f}")

    samples = {
        "Litwo_T1.0": generate_text(model, vocab, "Litwo, ojczyzno moja",
                                     length=400, temperature=1.0, device=device),
        "Litwo_T0.7": generate_text(model, vocab, "Litwo, ojczyzno moja",
                                     length=400, temperature=0.7, device=device),
    }
    save_run(name=tag,
             hyperparams={"block_size": block, "n_embd": embd, "n_head": head,
                          "n_layer": layer, "dropout": dropout, "lr": lr,
                          "batch_size": batch, "max_iters": iters,
                          "n_params": n_params},
             train_losses=tr, val_losses=vl, iters=iters_log, samples=samples,
             extra={"train_time_s": train_time, "best_val": best_val})
    return {"tag": tag, "block_size": block, "n_embd": embd, "n_head": head,
            "n_layer": layer, "n_params": n_params,
            "final_train": tr[-1], "final_val": vl[-1], "best_val": best_val,
            "time_s": train_time}


def main() -> None:
    device = pick_device()
    print(f"Device: {device}")
    text = load_text()
    vocab = build_vocab(text)
    data = encode(text, vocab)
    train_data, test_data = train_test_split(data)

    results: list[dict] = []
    for cfg in SPEED_CONFIGS:
        results.append(run_one(*cfg, device=device,
                               train_data=train_data, test_data=test_data, vocab=vocab))

    print("\nSpeed runs summary:")
    print(f"{'tag':>10} {'params':>9} {'time':>8}  {'train':>8}  {'val':>8}  {'best_val':>9}")
    for r in results:
        print(f"{r['tag']:>10} {r['n_params']:>9,} {r['time_s']/60:>5.1f}min  "
              f"{r['final_train']:>8.4f}  {r['final_val']:>8.4f}  {r['best_val']:>9.4f}")

    import json
    with open(LOSSES_DIR / "07_speed_runs_sweep.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
