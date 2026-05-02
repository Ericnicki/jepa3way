"""
TriadicFuser: one transformer that consumes [SHARED_Q | L | V | A] tokens.

The same trunk is invoked three times per training step, once per branch
(LV / VA / LA). The branch identity is encoded by zeroing the modality
*not* in the branch name and tagging every slot with a learned 4-way type
embedding (query / L / V / A) so the fuser can address a slot even when its
content is zero. The fuser reads out the SHARED_Q slot and projects it to
``d_shared``.

The output LayerNorm has ``elementwise_affine=False`` on purpose: a learnable
beta would be an invisible escape hatch against VICReg's per-dim variance
hinge (VICReg centers the batch before measuring variance, so a constant
shift is invisible to it and would let cosine similarity collapse to ~1
while per-dim std stays >= 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TriadicFuser(nn.Module):
    def __init__(self, d_model: int = 384, d_shared: int = 512,
                 n_heads: int = 6, ffn: int = 1536, n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.shared = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.shared, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(d_model, d_shared)
        self.ln_out = nn.LayerNorm(d_shared, elementwise_affine=False)

    def forward(self, l: torch.Tensor, v: torch.Tensor, a: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        l, v, a: (B, Nl, D), (B, Nv, D), (B, Na, D)
        key_padding_mask (optional): (B, 1 + Nl + Nv + Na) bool, True at
            positions to mask. Position 0 (the shared query) must be False.
        Returns: (B, d_shared).
        """
        B = l.shape[0]
        q = self.shared.expand(B, -1, -1)
        x = torch.cat([q, l, v, a], dim=1)
        x = self.enc(x, src_key_padding_mask=key_padding_mask)
        return self.ln_out(self.out(x[:, 0]))
