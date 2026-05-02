"""
VICReg variance + covariance hinge, applied per-tensor on the pre-projector
fuser output ``z`` (one call per branch).

eps=1e-3 is bf16-safe.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def vicreg_loss(z: torch.Tensor, var_coef: float = 25.0, cov_coef: float = 1.0,
                eps: float = 1e-3) -> torch.Tensor:
    # z: (B, D)
    B, D = z.shape
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(1.0 - std).mean()
    cov = (z.T @ z) / max(B - 1, 1)
    off = cov - torch.diag(torch.diagonal(cov))
    cov_loss = off.pow(2).sum() / D
    return var_coef * var_loss + cov_coef * cov_loss
