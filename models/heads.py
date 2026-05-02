"""
BYOL-style projector ``g`` and predictor ``h``.

Both use local BatchNorm1d (never SyncBN): the consensus alignment loss
relies on local-batch statistics as an implicit regularizer, and the
predictor has no EMA so there is no stale-BN interaction to worry about.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _MLPBNReLU(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.bn1 = nn.BatchNorm1d(d_hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.bn1(self.fc1(x))))


class Projector(_MLPBNReLU):
    pass


class Predictor(_MLPBNReLU):
    pass
