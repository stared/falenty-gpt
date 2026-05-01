"""Common NN training utilities used by 02..05 scripts."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import get_batch


@dataclass
class TrainConfig:
    block_size: int
    batch_size: int
    lr: float
    max_iters: int
    eval_every: int
    eval_iters: int = 20


def loss_fn(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy that handles both (B, V) and (B, T, V) outputs."""
    logits = model(x)
    if logits.dim() == 3:
        B, T, V = logits.shape
        return F.cross_entropy(logits.reshape(B*T, V), y.reshape(B*T))
    elif logits.dim() == 2:
        # (B, V), y: (B,)
        return F.cross_entropy(logits, y)
    else:
        raise ValueError(logits.shape)


@torch.no_grad()
def estimate_loss(model: nn.Module, train_data: torch.Tensor, test_data: torch.Tensor,
                  cfg: TrainConfig, device: str,
                  make_batch: Callable | None = None) -> dict[str, float]:
    """Estimate train/val loss by averaging over `eval_iters` batches."""
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", test_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            if make_batch is not None:
                xb, yb = make_batch(data, cfg.block_size, cfg.batch_size, device)
            else:
                xb, yb = get_batch(data, cfg.block_size, cfg.batch_size, device)
            losses[k] = loss_fn(model, xb, yb).item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model: nn.Module, train_data: torch.Tensor, test_data: torch.Tensor,
          cfg: TrainConfig, device: str,
          make_batch: Callable | None = None,
          weight_decay: float = 0.0,
          verbose: bool = True) -> tuple[list[int], list[float], list[float]]:
    """Standard training loop. Returns (iters, train_losses, val_losses)."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=weight_decay)

    iters: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    t0 = time.time()
    for it in range(cfg.max_iters):
        if make_batch is not None:
            xb, yb = make_batch(train_data, cfg.block_size, cfg.batch_size, device)
        else:
            xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        loss = loss_fn(model, xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it % cfg.eval_every == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(model, train_data, test_data, cfg, device, make_batch)
            iters.append(it)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            if verbose:
                elapsed = time.time() - t0
                print(f"  it {it:>5d}  train {losses['train']:.4f}  val {losses['val']:.4f}  ({elapsed:.1f}s)")
    return iters, train_losses, val_losses


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
