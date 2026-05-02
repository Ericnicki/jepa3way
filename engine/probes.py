"""
Diagnostic probes computed periodically during training.

All probes are forward-only on the *online* model (not the EMA target) and
report cosine-similarity-style scalars; nothing here ever aborts the run.

Probe summary:
    p1 (intra_diversity)        : per-branch within-batch cosine sim. 1.0 = collapsed.
    p2 (cross_mean_distinctness): cosine sim between branch means. High = each branch
                                  has a constant offset that survives mean-subtraction.
    p3 (lang_sensitivity)       : cosine drift on z_lv when L is shuffled across batch.
    p4 (per_sample_consensus)   : per-sample cosine sim averaged over branch pairs.
                                  HIGH = consensus achieved (this is the goal, not collapse).
    p5_null                     : cosine drift when L is replaced by the null token.
    p5_ratio = p3 / p5_null     : >> 1 means swapping languages is closer than removing
                                  it -> evidence of semantic grounding.
    p6_para                     : cosine drift between two paraphrases of the same
                                  instruction. Small = paraphrase-invariant.
    p7_swap_null                : cosine drift between (swapped L) and (null L).
    *swap_*                     : per-branch sensitivity to swapping L / V / A / proprio.
                                  Tells you which modality each branch is actually using.
    cos_*_*                     : pairwise inter-branch cosine. Proxy for how close
                                  the three z's already agree.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def intra_diversity(z: torch.Tensor) -> float:
    idx = torch.randperm(z.shape[0], device=z.device)
    a = F.normalize(z, dim=-1, eps=1e-6)
    b = F.normalize(z[idx], dim=-1, eps=1e-6)
    return (a * b).sum(-1).mean().item()


def cross_mean_distinctness(zs: list[torch.Tensor]) -> list[float]:
    means = [F.normalize(z.mean(0, keepdim=True), dim=-1, eps=1e-6) for z in zs]
    vals = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            vals.append((means[i] * means[j]).sum(-1).item())
    return vals


def cosine_drift(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    """1 - cos(z_a, z_b) averaged over the batch. Used by every *swap probe."""
    a = F.normalize(z_a, dim=-1, eps=1e-6)
    b = F.normalize(z_b, dim=-1, eps=1e-6)
    return 1.0 - (a * b).sum(-1).mean().item()


def per_sample_consensus(zs: list[torch.Tensor]) -> float:
    """Per-sample cosine sim averaged over all branch pairs.

    1.0 = per-sample modality-invariance achieved. This is the architectural
    goal, not a collapse signal -- the three branches consume disjoint
    modality views, so a high value here means the fuser produces a single
    consensus S regardless of which view it sees.
    """
    normed = [F.normalize(z, dim=-1, eps=1e-6) for z in zs]
    sims: list[torch.Tensor] = []
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            sims.append((normed[i] * normed[j]).sum(-1))
    return torch.stack(sims, dim=0).mean().item()
