"""Char-level Markov n-gram model.

Sweep over state size 1..8.
- Train on first 90% of Pan Tadeusz, evaluate cross-entropy on last 10%.
- Laplace (add-alpha) smoothing for unseen continuations.
- Back-off to shorter context if the full state has never been seen.

Loss reported as cross-entropy in nats per char (matches what NN models report).
"""
from __future__ import annotations

import math
import random
import time
from collections import Counter, defaultdict

from data_utils import (
    LOSSES_DIR, SAMPLES_DIR, SEED, build_vocab, load_text,
    save_run, train_test_split_text, random_baseline_loss, unigram_baseline_loss,
)

ALPHA = 0.1  # Laplace smoothing


def build_models(text: str, max_n: int) -> tuple[dict[int, dict[str, Counter]], Counter]:
    """For each state size 1..max_n, build dict[state_str -> Counter(next_char -> count)].
    Also return unigram counter (used as backoff)."""
    models: dict[int, dict[str, Counter]] = {n: defaultdict(Counter) for n in range(1, max_n + 1)}
    unigram = Counter(text)
    for n in range(1, max_n + 1):
        for i in range(len(text) - n):
            state = text[i:i+n]
            nxt = text[i+n]
            models[n][state][nxt] += 1
    return models, unigram


def conditional_logprob(model: dict[str, Counter], unigram: Counter, vocab_size: int,
                        state: str, ch: str) -> float:
    """log P(ch | state) with Laplace smoothing; backoff to unigram if state unseen."""
    if state in model:
        cnt = model[state]
        total = sum(cnt.values())
        p = (cnt.get(ch, 0) + ALPHA) / (total + ALPHA * vocab_size)
    else:
        # Unigram backoff (with smoothing)
        total_u = sum(unigram.values())
        p = (unigram.get(ch, 0) + ALPHA) / (total_u + ALPHA * vocab_size)
    return math.log(p)


def evaluate(models: dict[int, dict[str, Counter]], unigram: Counter, vocab_size: int,
             text: str, n: int) -> float:
    """Cross-entropy in nats per char on `text` using model of state size n."""
    if n <= 0 or n > max(models):
        raise ValueError(n)
    model = models[n]
    total_neg_logp = 0.0
    count = 0
    for i in range(n, len(text)):
        state = text[i-n:i]
        ch = text[i]
        logp = conditional_logprob(model, unigram, vocab_size, state, ch)
        total_neg_logp += -logp
        count += 1
    return total_neg_logp / max(count, 1)


def generate(model: dict[str, Counter], unigram: Counter, vocab_size: int,
             start_state: str, n: int, length: int, rng: random.Random) -> str:
    """Greedy-stochastic char generation with Laplace smoothing."""
    if len(start_state) < n:
        # pad with the most common char (' ')
        start_state = (" " * (n - len(start_state))) + start_state
    state = start_state[-n:]
    out = list(start_state)
    chars = sorted(unigram.keys())
    for _ in range(length):
        if state in model:
            cnt = model[state]
            total = sum(cnt.values())
            weights = [(cnt.get(c, 0) + ALPHA) / (total + ALPHA * vocab_size) for c in chars]
        else:
            total_u = sum(unigram.values())
            weights = [(unigram.get(c, 0) + ALPHA) / (total_u + ALPHA * vocab_size) for c in chars]
        next_ch = rng.choices(chars, weights=weights)[0]
        out.append(next_ch)
        state = "".join(out[-n:])
    return "".join(out)


def main() -> None:
    rng = random.Random(SEED)
    text = load_text()
    vocab = build_vocab(text)
    train_text, test_text = train_test_split_text(text)

    print(f"Vocab: {vocab.size}, train={len(train_text):,}, test={len(test_text):,}")
    print(f"Random baseline: {random_baseline_loss(vocab.size):.4f}")
    print(f"Unigram baseline: {unigram_baseline_loss(text, vocab):.4f}")
    print()

    max_n = 8
    print(f"Building Markov models for n=1..{max_n} ...")
    t0 = time.time()
    models, unigram_train = build_models(train_text, max_n)
    print(f"Built in {time.time() - t0:.2f}s")
    print()

    sweep_summary: list[dict] = []

    for n in range(1, max_n + 1):
        t0 = time.time()
        train_loss = evaluate(models, unigram_train, vocab.size, train_text, n)
        val_loss = evaluate(models, unigram_train, vocab.size, test_text, n)
        eval_time = time.time() - t0

        # Generate samples
        samples = {
            "Litwo": generate(models[n], unigram_train, vocab.size,
                              "Litwo, ojczyzno moja", n=n, length=400, rng=rng),
            "blank": generate(models[n], unigram_train, vocab.size,
                              " ", n=n, length=400, rng=rng),
        }

        # Vocabulary coverage
        n_states = len(models[n])
        avg_continuations = sum(len(c) for c in models[n].values()) / max(n_states, 1)

        print(f"[n={n}] states={n_states:>7,d}  avg_cont={avg_continuations:.2f}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  ({eval_time:.1f}s)")

        save_run(
            name=f"01_markov_n{n}",
            hyperparams={"state_size": n, "alpha": ALPHA},
            train_losses=[train_loss],
            val_losses=[val_loss],
            iters=[0],
            samples=samples,
            extra={
                "n_states": n_states,
                "avg_continuations": avg_continuations,
                "eval_time_s": eval_time,
            },
        )
        sweep_summary.append({
            "n": n,
            "n_states": n_states,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

    # Quick text overview of sweep
    print("\nSweep summary:")
    print(f"{'n':>2}  {'states':>10}  {'train':>8}  {'val':>8}  gap")
    for row in sweep_summary:
        gap = row["val_loss"] - row["train_loss"]
        print(f"{row['n']:>2}  {row['n_states']:>10,}  {row['train_loss']:>8.4f}  {row['val_loss']:>8.4f}  {gap:>+.4f}")

    # Save aggregate
    import json
    with open(LOSSES_DIR / "01_markov_sweep.json", "w", encoding="utf-8") as f:
        json.dump(sweep_summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
