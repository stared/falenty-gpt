"""Generate a side-by-side comparison of generated text from all best models.

Output: results/SAMPLES.md  - human-readable comparison grouped by prompt.
"""
from __future__ import annotations

import json
from pathlib import Path

from data_utils import LOSSES_DIR, RESULTS_DIR


def load_run(name: str):
    p = LOSSES_DIR / f"{name}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def best_run_name(sweep_name: str, model_kind: str) -> str | None:
    p = LOSSES_DIR / f"{sweep_name}.json"
    if not p.exists():
        return None
    with open(p) as f:
        rows = json.load(f)
    if not rows:
        return None
    best = min(rows, key=lambda r: r.get("best_val", r.get("val_loss", 1e9)))
    if model_kind == "linear":
        return f"02_linear_ctx{best['context_size']}_emb{best['embedding_dim']}"
    if model_kind == "mlp":
        return f"03_mlp_ctx{best['context_size']}_emb{best['embedding_dim']}_hid{best['hidden_dim']}"
    if model_kind == "single":
        return f"04_singlehead_b{best['block_size']}_e{best['n_embd']}_h{best['head_size']}"
    if model_kind == "multi":
        return f"05_minigpt_b{best['block_size']}_e{best['n_embd']}_h{best['n_head']}_l{best['n_layer']}"
    return None


def main() -> None:
    out: list[str] = []
    out.append("# Próbki tekstu - porównanie modeli\n")
    out.append("Wszystkie próbki rozpoczęte tym samym ciągiem `\"Litwo, ojczyzno moja\"`. "
               "Pokazujemy najlepszą konfigurację każdego modelu (z najmniejszym val loss).\n")
    out.append("Porządek: od najprostszego do najbardziej zaawansowanego.\n")

    # Define what to show
    items: list[tuple[str, str]] = []
    if (LOSSES_DIR / "01_markov_n3.json").exists():
        items.append(("Markov n=1 (bigram)", "01_markov_n1"))
        items.append(("Markov n=3 (sweet spot)", "01_markov_n3"))
        items.append(("Markov n=5 (overfit, z back-off)", "01_markov_n5"))
        items.append(("Markov n=8 (memorizuje fragmenty)", "01_markov_n8"))

    name = best_run_name("02_linear_sweep", "linear")
    if name:
        items.append((f"Linear (best: {name})", name))
    name = best_run_name("03_mlp_sweep", "mlp")
    if name:
        items.append((f"MLP (best: {name})", name))
    name = best_run_name("04_singlehead_sweep", "single")
    if name:
        items.append((f"Transformer 1 head (best: {name})", name))
    name = best_run_name("05_multihead_sweep", "multi")
    if name:
        items.append((f"Mini-GPT (best: {name})", name))

    extended_1p2m = LOSSES_DIR / "06_extended_1p2M.json"
    if extended_1p2m.exists():
        items.append(("Mini-GPT EXTENDED 1.2M (10k iter)", "06_extended_1p2M"))
    extended_2p7m = LOSSES_DIR / "06_extended_2p7M.json"
    if extended_2p7m.exists():
        items.append(("Mini-GPT EXTENDED 2.7M (10k iter)", "06_extended_2p7M"))

    # Show each prompt-group
    prompts = [("Litwo_T0.7", "T=0.7"), ("Litwo_T1.0", "T=1.0")]
    for prompt_key, prompt_label in prompts:
        out.append(f"\n## Litwo, ojczyzno moja… ({prompt_label})\n")
        for label, name in items:
            d = load_run(name)
            if not d:
                continue
            samples = d.get("samples", {})
            # Some markov uses "Litwo" key only
            key = prompt_key
            if key not in samples:
                key = "Litwo"
            sample = samples.get(key, "")
            if not sample:
                continue
            val = None
            if d.get("val_losses"):
                val = min(d["val_losses"])
            label_with_val = f"{label}" + (f" — val={val:.3f}" if val else "")
            out.append(f"### {label_with_val}\n```\n{sample[:600].rstrip()}\n```\n")

    text = "\n".join(out)
    out_path = RESULTS_DIR / "SAMPLES.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
