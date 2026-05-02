"""EMA tau and learning-rate schedules."""
from __future__ import annotations

import math


def cosine_tau(step: int, total: int, start: float = 0.99, end: float = 0.9995) -> float:
    """EMA momentum: cosine ramp from `start` (early) to `end` (late)."""
    t = min(step / max(total, 1), 1.0)
    return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * t))


def lr_with_warmup_cosine(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
