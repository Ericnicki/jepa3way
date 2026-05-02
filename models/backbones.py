"""
Frozen language and vision encoders, plus the trainable action token embedder.

The language and vision backbones are loaded once at precompute time
(``scripts/precompute_features.py``) and the resulting per-episode tensors are
read from disk by the trainer. The encoder classes themselves are kept here
so the precompute script and the trainer share one source of truth for input
shapes and normalization conventions.

Output shapes:
    DinoV2Vision : (B, T_v, 256, 768)   -- T_v=1, 16x16 patch grid, CLS dropped
    T5Language   : (B, n_tok, 768)      -- mean-pooled across K paraphrases when K>1
    ActionTokenEmbed : (B, H, d_model)  -- H = action sequence length
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _freeze_and_eval(m: nn.Module) -> nn.Module:
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def sinusoidal_pos(n: int, d: int, device=None) -> torch.Tensor:
    pe = torch.zeros(n, d, device=device)
    pos = torch.arange(n, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, device=device).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class DinoV2Vision(nn.Module):
    """Frozen DINOv2-B/14 encoder.

    Input:  (B, T_v, 3, 224, 224), pixel values in [0, 1]. ImageNet mean/std
            normalization is applied internally so the dataset pipeline does
            not need to.
    Output: (B, T_v, 256, 768). The CLS token is dropped; 224/14 = 16 yields
            16*16 = 256 patch tokens at d=768.
    """

    def __init__(self, weights_dir: str, tokens_per_frame: int = 256):
        super().__init__()
        from transformers import Dinov2Model
        net = Dinov2Model.from_pretrained(weights_dir)
        self.net = _freeze_and_eval(net)
        self.tokens_per_frame = tokens_per_frame
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def train(self, mode: bool = True):  # type: ignore[override]
        # Force the frozen backbone to stay in eval mode regardless of outer
        # .train() calls, otherwise dropout layers inside the HF model would
        # silently re-activate after the first model.train().
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = vid.shape
        x = vid.reshape(B * T, C, H, W)
        x = (x - self.pixel_mean) / self.pixel_std
        with torch.no_grad():
            out = self.net(pixel_values=x)
        tokens = out.last_hidden_state[:, 1:, :]  # drop [CLS]
        N = tokens.shape[1]
        if N < self.tokens_per_frame:
            tokens = F.pad(tokens, (0, 0, 0, self.tokens_per_frame - N))
        elif N > self.tokens_per_frame:
            tokens = tokens[:, : self.tokens_per_frame, :]
        D = tokens.shape[-1]
        return tokens.reshape(B, T, self.tokens_per_frame, D)


class T5Language(nn.Module):
    """Frozen T5-base encoder.

    Input:  (B, n_tok) or (B, K, n_tok) int64 token ids. K>1 (multiple
            paraphrases) is encoded independently and mean-pooled along K.
    Output: (B, n_tok, 768). pad_id=0; attention_mask is derived from
            ``ids != pad_id`` so pad positions do not bleed into others.
    """

    def __init__(self, weights_dir: str, n_tok: int = 32, pad_id: int = 0):
        super().__init__()
        from transformers import T5EncoderModel
        net = T5EncoderModel.from_pretrained(weights_dir)
        self.net = _freeze_and_eval(net)
        self.n_tok = n_tok
        self.pad_id = pad_id

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dim() == 3:
            B, K, L = ids.shape
            ids = ids.reshape(B * K, L)
            out = self._encode(ids)
            D = out.shape[-1]
            return out.reshape(B, K, self.n_tok, D).mean(dim=1)
        return self._encode(ids)

    def _encode(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        if L < self.n_tok:
            ids = F.pad(ids, (0, self.n_tok - L), value=self.pad_id)
        elif L > self.n_tok:
            ids = ids[:, : self.n_tok]
        attn = (ids != self.pad_id).long()
        with torch.no_grad():
            out = self.net(input_ids=ids, attention_mask=attn)
        return out.last_hidden_state


class ActionTokenEmbed(nn.Module):
    """14-slot canonical action -> per-step d_model token, optional proprio sum.

    The action input is the (a - mu)/sigma normalized vector concatenated with
    its slot-validity mask, so feats has shape (B, H, 2 * action_slots).
    Sinusoidal position codes over H let the fuser tell timesteps apart.

    When ``proprio_dim > 0`` and a proprio tensor is supplied at forward time,
    a parallel Linear projects (B, H, proprio_dim) into d_model and is summed
    onto the action embedding -- the per-step "A" view becomes a state-action
    trajectory rather than action only.
    """

    def __init__(self, action_slots: int, d_model: int, proprio_dim: int = 0):
        super().__init__()
        self.proj = nn.Linear(2 * action_slots, d_model)
        self.proj_p = nn.Linear(proprio_dim, d_model) if proprio_dim > 0 else None
        self.register_buffer("pos", sinusoidal_pos(256, d_model), persistent=False)

    def forward(self, feats: torch.Tensor,
                prop: torch.Tensor = None) -> torch.Tensor:
        # feats: (B, H, 2 * action_slots). prop (optional): (B, H, proprio_dim).
        x = self.proj(feats)
        H = x.shape[1]
        x = x + self.pos[:H].unsqueeze(0)
        if self.proj_p is not None and prop is not None:
            x = x + self.proj_p(prop)
        return x
