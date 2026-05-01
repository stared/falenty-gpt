"""Markov with stupid back-off for evaluation.

The plain Markov sweep treats unseen states by falling back to unigram, which
hurts large-n test loss. With back-off (try n, then n-1, ..., then unigram),
larger n should never do worse than smaller n on test data.

This script computes back-off val loss for each n and saves it as a separate
sweep file (does not overwrite the original).
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

from data_utils import (
    LOSSES_DIR, SEED, build_vocab, load_text, train_test_split_text,
)

ALPHA = 0.1


def build_models(text: str, max_n: int):
    models = {n: defaultdict(Counter) for n in range(1, max_n + 1)}
    for n in range(1, max_n + 1):
        for i in range(len(text) - n):
            state = text[i:i+n]
            nxt = text[i+n]
            models[n][state][nxt] += 1
    unigram = Counter(text)
    return models, unigram


def evaluate_backoff(models, unigram, vocab_size, text, n) -> float:
    """Cross-entropy with back-off: try n, then n-1, ..., then unigram."""
    total_neg = 0.0
    count = 0
    for i in range(n, len(text)):
        ch = text[i]
        logp = None
        for k in range(n, 0, -1):
            state = text[i-k:i]
            model_k = models[k]
            if state in model_k:
                cnt = model_k[state]
                total = sum(cnt.values())
                p = (cnt.get(ch, 0) + ALPHA) / (total + ALPHA * vocab_size)
                logp = math.log(p)
                break
        if logp is None:
            total_u = sum(unigram.values())
            p = (unigram.get(ch, 0) + ALPHA) / (total_u + ALPHA * vocab_size)
            logp = math.log(p)
        total_neg += -logp
        count += 1
    return total_neg / max(count, 1)


def main() -> None:
    text = load_text()
    vocab = build_vocab(text)
    train_text, test_text = train_test_split_text(text)

    max_n = 8
    print(f"Building models for n=1..{max_n} ...")
    models, unigram_train = build_models(train_text, max_n)

    rows = []
    for n in range(1, max_n + 1):
        train_loss = evaluate_backoff(models, unigram_train, vocab.size, train_text, n)
        val_loss = evaluate_backoff(models, unigram_train, vocab.size, test_text, n)
        gap = val_loss - train_loss
        n_states = len(models[n])
        print(f"[n={n}] states={n_states:>7,d}  train={train_loss:.4f}  val={val_loss:.4f}  gap={gap:+.4f}")
        rows.append({"n": n, "n_states": n_states,
                     "train_loss": train_loss, "val_loss": val_loss})

    with open(LOSSES_DIR / "01b_markov_backoff_sweep.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
