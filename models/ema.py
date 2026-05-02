"""
EMA target wrapper.

Use ``ema_forward(ema_mod)`` as a context manager rather than calling
``ema_mod.target(*args)`` directly: the context guarantees ``eval()`` and
``no_grad`` regardless of caller state, which is what every existing
training and probe path assumes.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
import torch.nn as nn


class EMAWrapper(nn.Module):
    def __init__(self, online: nn.Module):
        super().__init__()
        self.target = copy.deepcopy(online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    @torch.no_grad()
    def update(self, online: nn.Module, tau: float) -> None:
        for pt, po in zip(self.target.parameters(), online.parameters()):
            pt.data.mul_(tau).add_(po.data, alpha=1.0 - tau)
        for bt, bo in zip(self.target.buffers(), online.buffers()):
            if bt.dtype == bo.dtype and bt.shape == bo.shape and bt.is_floating_point():
                bt.data.mul_(tau).add_(bo.data, alpha=1.0 - tau)

    def forward(self, *args, **kwargs):
        self.target.eval()
        with torch.no_grad():
            return self.target(*args, **kwargs)


@contextmanager
def ema_forward(ema: EMAWrapper):
    ema.target.eval()
    with torch.no_grad():
        yield ema.target
