"""Print a quick status of all sweep results so far."""
from __future__ import annotations

import json
from pathlib import Path

from data_utils import LOSSES_DIR


def load(name: str):
    p = LOSSES_DIR / f"{name}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def show(name: str, rows, key_loss: str = "best_val"):
    if not rows:
        print(f"-- {name}: no data --")
        return
    rows = sorted(rows, key=lambda r: r.get(key_loss, r.get("val_loss", 1e9)))
    print(f"\n=== {name} ({len(rows)} configs) ===")
    print(f"  best val: {rows[0].get(key_loss, rows[0].get('val_loss')):.4f}")
    for r in rows[:5]:
        cfg = ", ".join(f"{k}={v}" for k, v in r.items()
                        if k not in ("train_loss", "val_loss", "best_val", "final_train",
                                     "final_val", "n_params", "n_states", "time_s"))
        nstates_or_params = r.get("n_states", r.get("n_params"))
        nstr = f"{nstates_or_params:,}" if nstates_or_params else "?"
        print(f"  best_val={r.get(key_loss, r.get('val_loss')):.4f} "
              f"({nstr} params) {cfg}")


def main() -> None:
    show("Markov", load("01_markov_sweep"), "val_loss")
    show("Markov (back-off)", load("01b_markov_backoff_sweep"), "val_loss")
    show("Linear", load("02_linear_sweep"))
    show("MLP", load("03_mlp_sweep"))
    show("Single-head transformer", load("04_singlehead_sweep"))
    show("Mini-GPT (multi-head)", load("05_multihead_sweep"))


if __name__ == "__main__":
    main()
