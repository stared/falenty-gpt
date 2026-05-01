"""Shared data utilities used by all model scripts.

Same train/test split (90/10) and same data (Pan Tadeusz, char-level)
across all five experiments.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

import torch

DATA_PATH = Path(__file__).parent.parent / "data" / "pan-tadeusz.txt"
RESULTS_DIR = Path(__file__).parent / "results"
LOSSES_DIR = RESULTS_DIR / "losses"
SAMPLES_DIR = RESULTS_DIR / "samples"

TRAIN_FRACTION = 0.9
SEED = 42


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Vocab:
    chars: list[str]
    char2id: dict[str, int]
    id2char: dict[int, str]
    size: int


def load_text() -> str:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str) -> Vocab:
    chars = sorted(list(set(text)))
    char2id = {c: i for i, c in enumerate(chars)}
    id2char = {i: c for i, c in enumerate(chars)}
    return Vocab(chars=chars, char2id=char2id, id2char=id2char, size=len(chars))


def encode(text: str, vocab: Vocab) -> torch.Tensor:
    return torch.tensor([vocab.char2id[c] for c in text], dtype=torch.long)


def decode(ids: list[int], vocab: Vocab) -> str:
    return "".join(vocab.id2char[i] for i in ids)


def train_test_split(data: torch.Tensor, fraction: float = TRAIN_FRACTION) -> tuple[torch.Tensor, torch.Tensor]:
    split = int(len(data) * fraction)
    return data[:split], data[split:]


def train_test_split_text(text: str, fraction: float = TRAIN_FRACTION) -> tuple[str, str]:
    split = int(len(text) * fraction)
    return text[:split], text[split:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int,
              device: str, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Random batch of (x, y) pairs of length block_size each.

    y[i] is x[i] shifted by one - we predict the next character at every position.
    """
    if generator is None:
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    else:
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


def save_run(name: str, hyperparams: dict, train_losses: list[float],
             val_losses: list[float], iters: list[int], samples: dict[str, str],
             extra: dict | None = None) -> None:
    """Save a run's metrics + samples to JSON for later analysis."""
    LOSSES_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "hyperparams": hyperparams,
        "iters": iters,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "samples": samples,
        "extra": extra or {},
    }
    out = LOSSES_DIR / f"{name}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    sample_txt = SAMPLES_DIR / f"{name}.txt"
    with open(sample_txt, "w", encoding="utf-8") as f:
        f.write(f"# {name}\n\n")
        f.write(f"## hyperparams\n```\n{json.dumps(hyperparams, ensure_ascii=False, indent=2)}\n```\n\n")
        if extra:
            f.write(f"## extra\n```\n{json.dumps(extra, ensure_ascii=False, indent=2)}\n```\n\n")
        f.write(f"## final losses\n")
        if train_losses:
            f.write(f"- train: {train_losses[-1]:.4f}\n")
        if val_losses:
            f.write(f"- val: {val_losses[-1]:.4f}\n")
        f.write("\n")
        for sname, stext in samples.items():
            f.write(f"## sample [{sname}]\n```\n{stext}\n```\n\n")


def random_baseline_loss(vocab_size: int) -> float:
    """Loss of a uniform random model (upper bound for any sane model)."""
    return math.log(vocab_size)


def unigram_baseline_loss(text: str, vocab: Vocab) -> float:
    """Loss of a model that always outputs the marginal char distribution."""
    from collections import Counter
    counts = Counter(text)
    total = sum(counts.values())
    loss = 0.0
    for c, k in counts.items():
        p = k / total
        loss += -p * math.log(p)
    return loss


if __name__ == "__main__":
    text = load_text()
    vocab = build_vocab(text)
    print(f"Text length: {len(text):,} characters")
    print(f"Vocab size: {vocab.size}")
    print(f"Chars: {''.join(vocab.chars)!r}")
    train_text, test_text = train_test_split_text(text)
    print(f"Train: {len(train_text):,}  Test: {len(test_text):,}")
    print(f"Random baseline loss: {random_baseline_loss(vocab.size):.4f}")
    print(f"Unigram baseline loss: {unigram_baseline_loss(text, vocab):.4f}")
    print(f"Device: {pick_device()}")
