"""Generate loss curve plots from the saved JSON results."""
from __future__ import annotations

import json
import os
from pathlib import Path

# Set matplotlib config dir somewhere writable
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.makedirs("/tmp/matplotlib", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import LOSSES_DIR, RESULTS_DIR


def load(name: str):
    path = LOSSES_DIR / f"{name}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def best_run(sweep_path: Path, key: str = "best_val") -> str | None:
    if not sweep_path.exists():
        return None
    with open(sweep_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        return None
    best = min(rows, key=lambda r: r.get(key, r.get("val_loss", 1e9)))
    if "context_size" in best and "embedding_dim" in best and "hidden_dim" in best:
        return f"03_mlp_ctx{best['context_size']}_emb{best['embedding_dim']}_hid{best['hidden_dim']}"
    if "context_size" in best and "embedding_dim" in best:
        return f"02_linear_ctx{best['context_size']}_emb{best['embedding_dim']}"
    if "n_layer" in best:
        return f"05_minigpt_b{best['block_size']}_e{best['n_embd']}_h{best['n_head']}_l{best['n_layer']}"
    if "head_size" in best:
        return f"04_singlehead_b{best['block_size']}_e{best['n_embd']}_h{best['head_size']}"
    return None


def plot_loss_curves():
    """Plot best run from each model on a single chart."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    runs = []
    for sweep_name in ["02_linear_sweep", "03_mlp_sweep", "04_singlehead_sweep", "05_multihead_sweep"]:
        path = LOSSES_DIR / f"{sweep_name}.json"
        run_name = best_run(path)
        if run_name:
            d = load(run_name)
            if d:
                label_map = {
                    "02_linear": "Linear",
                    "03_mlp": "MLP",
                    "04_single": "Transformer (1 head)",
                    "05_minigpt": "Mini-GPT",
                }
                label = next((v for k, v in label_map.items() if k in run_name), run_name)
                runs.append((label, d, run_name))

    # Plot training curves (left) and val curves (right)
    for label, d, name in runs:
        iters = d["iters"]
        ax[0].plot(iters, d["train_losses"], label=f"{label}", linewidth=1.5)
        ax[1].plot(iters, d["val_losses"], label=f"{label}", linewidth=1.5)

    # Add extended runs
    for ext_name, label, color in [
        ("06_extended_1p2M", "Mini-GPT 1.2M ext (10k iter)", "C5"),
        ("06_extended_2p7M", "Mini-GPT 2.7M ext (10k iter)", "C6"),
    ]:
        d = load(ext_name)
        if d:
            iters = d["iters"]
            ax[0].plot(iters, d["train_losses"], label=label, linewidth=1.5, color=color, linestyle="--")
            ax[1].plot(iters, d["val_losses"], label=label, linewidth=1.5, color=color, linestyle="--")

    # Add Markov baselines (horizontal lines for best n=3)
    markov = load("01_markov_n3")
    if markov and markov["val_losses"]:
        ax[1].axhline(y=markov["val_losses"][0], color="gray", linestyle="--",
                      alpha=0.7, label="Markov n=3 (val)")

    for a, title in zip(ax, ["train loss", "val loss"]):
        a.set_xlabel("iteration")
        a.set_ylabel("cross-entropy (nats/char)")
        a.set_title(title)
        a.grid(True, alpha=0.3)
        a.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out = RESULTS_DIR / "loss_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


def plot_sweep_summary():
    """Show val loss vs # parameters for each model family."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"markov": "C0", "linear": "C1", "mlp": "C2",
              "single_head": "C3", "multi_head": "C4"}

    # Markov: n_states vs val_loss
    markov_path = LOSSES_DIR / "01_markov_sweep.json"
    if markov_path.exists():
        with open(markov_path) as f:
            rows = json.load(f)
        ax.scatter([r["n_states"] for r in rows], [r["val_loss"] for r in rows],
                   c=colors["markov"], label="Markov (n_states)", s=80)
        for r in rows:
            ax.annotate(f"n={r['n']}", (r["n_states"], r["val_loss"]),
                        xytext=(5, 0), textcoords="offset points", fontsize=8)

    for sname, key, label in [
        ("02_linear_sweep", "n_params", "Linear"),
        ("03_mlp_sweep", "n_params", "MLP"),
        ("04_singlehead_sweep", "n_params", "Transformer (1 head)"),
        ("05_multihead_sweep", "n_params", "Mini-GPT"),
    ]:
        path = LOSSES_DIR / f"{sname}.json"
        if not path.exists():
            continue
        with open(path) as f:
            rows = json.load(f)
        x = [r[key] for r in rows]
        y = [r.get("best_val", r.get("val_loss")) for r in rows]
        col = colors.get(sname.split("_")[1], "k")
        ax.scatter(x, y, label=label, s=80, alpha=0.8)

    # Extended training (extra points with star marker)
    for ext_name, label, color in [
        ("06_extended_1p2M", "Mini-GPT 1.2M ext (10k iter)", "C5"),
        ("06_extended_2p7M", "Mini-GPT 2.7M ext (10k iter)", "C6"),
    ]:
        ext_path = LOSSES_DIR / f"{ext_name}.json"
        if ext_path.exists():
            with open(ext_path) as f:
                d = json.load(f)
            n_params = d.get("hyperparams", {}).get("n_params")
            best_val = min(d.get("val_losses", [float("inf")]))
            if n_params:
                ax.scatter([n_params], [best_val], marker="*", s=300,
                           c=color, edgecolors="black", linewidth=1.5,
                           label=label, zorder=5)

    # Baselines
    from data_utils import build_vocab, load_text, random_baseline_loss, unigram_baseline_loss
    text = load_text()
    vocab = build_vocab(text)
    ax.axhline(unigram_baseline_loss(text, vocab), color="black", linestyle=":",
               alpha=0.6, label=f"Unigram ({unigram_baseline_loss(text, vocab):.2f})")

    ax.set_xscale("log")
    ax.set_xlabel("# parameters / states (log scale)")
    ax.set_ylabel("validation cross-entropy (nats/char)")
    ax.set_title("Pan Tadeusz - char-level: val loss vs model size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "sweep_overview.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_loss_curves()
    plot_sweep_summary()
