"""
14-slot canonical action schema and per-dataset normalization registry.

Slots (fixed order):
    0..5   : ee_dx, ee_dy, ee_dz, ee_rx, ee_ry, ee_rz
    6      : gripper
    7..13  : j1..j7

Per-dataset stats (mu, sigma, mask) are loaded from the JSON file written by
``scripts/compute_droid_stats.py``. The path is read from the
``DROID_STATS_JSON`` env var by ``default_registry()``; the training entry
script sets it before constructing the dataset.

Slots whose sigma is below ``SIGMA_FLOOR`` are auto-masked and have their
sigma replaced by 1.0 to prevent division-by-zero.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

ACTION_SLOTS = 14
SIGMA_FLOOR = 1e-4


@dataclass
class DatasetActionSpec:
    name: str
    mu: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray

    def __post_init__(self):
        self.mu = np.asarray(self.mu, dtype=np.float32)
        self.sigma = np.asarray(self.sigma, dtype=np.float32)
        self.mask = np.asarray(self.mask, dtype=np.float32)
        deg = self.sigma < SIGMA_FLOOR
        self.mask = self.mask * (~deg).astype(np.float32)
        self.sigma = np.where(deg, 1.0, self.sigma).astype(np.float32)


class ActionRegistry:
    def __init__(self):
        self._specs: dict[str, DatasetActionSpec] = {}

    def register(self, spec: DatasetActionSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> DatasetActionSpec:
        return self._specs[name]

    def __contains__(self, name: str) -> bool:
        return name in self._specs


def load_stats_json(path: str) -> DatasetActionSpec:
    with open(path, "r") as f:
        d = json.load(f)
    return DatasetActionSpec(
        name=d["name"],
        mu=np.asarray(d["mu"], dtype=np.float32),
        sigma=np.asarray(d["sigma"], dtype=np.float32),
        mask=np.asarray(d["mask"], dtype=np.float32),
    )


def default_registry(droid_stats_path: str | None = None) -> ActionRegistry:
    """Return a registry holding the DROID spec.

    Stats path resolution: explicit arg > ``DROID_STATS_JSON`` env var > a
    safe stub (zero mu, sigma=0.25, all slots active). The stub exists only
    so unit tests that never touch real data can still construct a registry.
    """
    reg = ActionRegistry()
    if droid_stats_path is None:
        droid_stats_path = os.environ.get("DROID_STATS_JSON")
    if droid_stats_path and os.path.exists(droid_stats_path):
        reg.register(load_stats_json(droid_stats_path))
    else:
        droid_mu = np.zeros(ACTION_SLOTS, dtype=np.float32)
        droid_sigma = np.ones(ACTION_SLOTS, dtype=np.float32) * 0.25
        droid_mask = np.ones(ACTION_SLOTS, dtype=np.float32)
        reg.register(DatasetActionSpec("droid", droid_mu, droid_sigma, droid_mask))
    return reg


def tokenize_batch(actions: torch.Tensor, dataset_names: list[str],
                   registry: ActionRegistry) -> tuple[torch.Tensor, torch.Tensor]:
    """
    actions: (B, H, 14) raw values.
    Returns:
        feats: (B, H, 28) = concat((a - mu) / sigma * mask, mask)
        mask:  (B, H, 14) -- broadcast of the per-sample slot mask
    """
    B, H, S = actions.shape
    assert S == ACTION_SLOTS
    device = actions.device
    mu = torch.zeros(B, S, device=device)
    sig = torch.ones(B, S, device=device)
    m = torch.zeros(B, S, device=device)
    for i, name in enumerate(dataset_names):
        spec = registry.get(name)
        mu[i] = torch.from_numpy(spec.mu).to(device)
        sig[i] = torch.from_numpy(spec.sigma).to(device)
        m[i] = torch.from_numpy(spec.mask).to(device)
    mu_b = mu.unsqueeze(1)
    sig_b = sig.unsqueeze(1)
    m_b = m.unsqueeze(1).expand(B, H, S)
    normed = ((actions - mu_b) / sig_b) * m_b
    return torch.cat([normed, m_b], dim=-1), m_b
