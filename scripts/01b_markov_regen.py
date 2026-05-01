"""Regenerate Markov samples with sharper sampling (lower alpha) for readability.

The original 01_markov.py used ALPHA=0.1 for both training and sampling, which
makes generated text noisy because rare characters leak into outputs.

We keep the trained models intact (loss numbers don't change), but generate
new samples with alpha=0.001 to remove the noise. This is purely a
presentation step — does not affect any losses.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from data_utils import (
    LOSSES_DIR, SAMPLES_DIR, SEED, build_vocab, load_text, train_test_split_text,
)

ALPHA_GEN = 0.001


def build_models(text: str, max_n: int):
    models = {n: defaultdict(Counter) for n in range(1, max_n + 1)}
    for n in range(1, max_n + 1):
        for i in range(len(text) - n):
            state = text[i:i+n]
            nxt = text[i+n]
            models[n][state][nxt] += 1
    unigram = Counter(text)
    return models, unigram


def generate(models_by_n, unigram, vocab_size, start, n, length, rng):
    """Generate with stupid back-off: try n, then n-1, ..., then unigram."""
    if len(start) < n:
        start = (" " * (n - len(start))) + start
    out = list(start)
    chars = sorted(unigram.keys())
    for _ in range(length):
        weights = None
        # Back-off from n to 1
        for k in range(n, 0, -1):
            state = "".join(out[-k:])
            model_k = models_by_n[k]
            if state in model_k:
                cnt = model_k[state]
                total = sum(cnt.values())
                weights = [(cnt.get(c, 0) + ALPHA_GEN) / (total + ALPHA_GEN * vocab_size)
                           for c in chars]
                break
        if weights is None:
            total_u = sum(unigram.values())
            weights = [(unigram.get(c, 0) + ALPHA_GEN) / (total_u + ALPHA_GEN * vocab_size)
                       for c in chars]
        c = rng.choices(chars, weights=weights)[0]
        out.append(c)
    return "".join(out)


def main() -> None:
    rng = random.Random(SEED)
    text = load_text()
    vocab = build_vocab(text)
    train_text, _ = train_test_split_text(text)

    max_n = 8
    models, unigram = build_models(train_text, max_n)

    for n in range(1, max_n + 1):
        # Refresh samples in the saved JSON (preserves train/val losses)
        path = LOSSES_DIR / f"01_markov_n{n}.json"
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        new_samples = {
            "Litwo": generate(models, unigram, vocab.size,
                              "Litwo, ojczyzno moja", n, 400, rng),
            "blank": generate(models, unigram, vocab.size,
                              " ", n, 400, rng),
        }
        payload["samples"] = new_samples
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # Also rewrite the .txt sample file
        from data_utils import save_run
        save_run(
            name=f"01_markov_n{n}",
            hyperparams=payload["hyperparams"],
            train_losses=payload["train_losses"],
            val_losses=payload["val_losses"],
            iters=payload["iters"],
            samples=new_samples,
            extra=payload.get("extra"),
        )

        print(f"[n={n}] regenerated samples (alpha_gen={ALPHA_GEN})")
        print("  Litwo:", repr(new_samples["Litwo"][:120]))


if __name__ == "__main__":
    main()
