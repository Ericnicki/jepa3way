"""
Leave-one-out consensus alignment (BYOL-style).

For each online branch, the target is the L2-normalized mean of the *other
two* EMA projector outputs. Excluding a branch's own EMA from its target is
what keeps the BYOL predictor+EMA mechanism non-trivial: the LV predictor
cannot satisfy its target by echoing its own lagged copy -- it must produce
the average of what VA and LA see, and those two branches consume
structurally different inputs.

Targets are detached so gradients flow only through the online predictors.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def consensus_target_loo(p_lv_t: torch.Tensor, p_va_t: torch.Tensor,
                         p_la_t: torch.Tensor, eps: float = 1e-6):
    """Returns (t_lv, t_va, t_la), each L2-normalized and detached."""
    a = F.normalize(p_lv_t, dim=-1, eps=eps)
    b = F.normalize(p_va_t, dim=-1, eps=eps)
    c = F.normalize(p_la_t, dim=-1, eps=eps)
    t_lv = F.normalize((b + c) / 2.0, dim=-1, eps=eps).detach()
    t_va = F.normalize((a + c) / 2.0, dim=-1, eps=eps).detach()
    t_la = F.normalize((a + b) / 2.0, dim=-1, eps=eps).detach()
    return t_lv, t_va, t_la


def align_loss_loo(p_lv: torch.Tensor, p_va: torch.Tensor, p_la: torch.Tensor,
                   t_lv: torch.Tensor, t_va: torch.Tensor, t_la: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    q1 = F.normalize(p_lv, dim=-1, eps=eps)
    q2 = F.normalize(p_va, dim=-1, eps=eps)
    q3 = F.normalize(p_la, dim=-1, eps=eps)
    cos = (q1 * t_lv).sum(-1) + (q2 * t_va).sum(-1) + (q3 * t_la).sum(-1)
    return -(cos / 3.0).mean()
