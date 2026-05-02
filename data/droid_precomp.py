"""
Precomputed-feature DROID dataset (single-frame V + variable-length state-action
trajectory).

Per-episode files expected under ``feats_root/chunk-NNN/``:

    episode_XXXXXX_vision.npy       float16   (1, 256, 768)   -- DINOv2 patch tokens, frame 0
    episode_XXXXXX_lang.npy         float16   (3, 32, 768)    -- T5 encodings, 3 paraphrases
    episode_XXXXXX_actions_seq.npy  float32   (N_ep, 14)      -- actions at fixed wall-clock stride
    episode_XXXXXX_proprio_seq.npy  float32   (N_ep, 20)      -- proprio at the same stride

``N_ep`` is variable across episodes; ``__getitem__`` truncates each episode to
``max_seq_len`` and ``collate_precomp`` pads to the per-batch maximum, emitting
a ``seq_valid_mask`` so the trainer can mask padding positions in the fuser.

Paraphrase expansion (training-time, not precompute-time): each episode is
materialized as ``paraphrase_expansion`` distinct samples, one per paraphrase
slot. The default mean-pool over paraphrases (matching what the T5 backbone
does internally) is replaced by single-paraphrase selection so the model sees
genuinely different L tokens across the expanded copies. ``lang_feat_raw``
remains the full (3, 32, 768) tensor for the paraphrase-distance probe.

Optional quality filters (``filter_success``, ``filter_empty_L``) read the
per-episode parquet metadata at index-build time and drop episodes that fail
the requested checks. The filtered index is cached via ``index_cache`` so
subsequent launches skip the parquet rescan.

mmap is on by default to keep RAM bounded over the full ~38 GB feature set.
"""
from __future__ import annotations

import glob
import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DROIDPrecompConfig:
    feats_root: str
    chunks: list[str] = field(default_factory=list)  # empty == all chunk-* dirs present
    mmap: bool = True
    index_cache: Optional[str] = None
    max_seq_len: int = 64

    # Quality filters require the original DROID parquets; only consulted at
    # index-build time, not in __getitem__.
    droid_root: Optional[str] = None
    filter_success: bool = False
    filter_empty_L: bool = False

    paraphrase_expansion: int = 1


class DROIDPrecompDataset(Dataset):
    """One sample per (episode, paraphrase_slot) pair."""

    EMBODIMENT_ID = 0  # Franka

    def __init__(self, cfg: DROIDPrecompConfig, verbose: bool = True):
        self.cfg = cfg
        if not os.path.isdir(cfg.feats_root):
            raise FileNotFoundError(cfg.feats_root)
        self.items: list[tuple] = []
        self._build_index(verbose=verbose)

    # ---- index ------------------------------------------------------------

    def _scan_one(self, chunk_dir: str) -> list[tuple]:
        vision_paths = sorted(glob.glob(os.path.join(chunk_dir, "episode_*_vision.npy")))
        out: list[tuple] = []
        for vpath in vision_paths:
            base = os.path.basename(vpath)[: -len("_vision.npy")]
            lpath = os.path.join(chunk_dir, base + "_lang.npy")
            aseq = os.path.join(chunk_dir, base + "_actions_seq.npy")
            pseq = os.path.join(chunk_dir, base + "_proprio_seq.npy")
            if all(os.path.isfile(p) for p in (lpath, aseq, pseq)):
                out.append((vpath, lpath, aseq, pseq))
        return out

    def _build_index(self, verbose: bool) -> None:
        if self.cfg.index_cache and os.path.isfile(self.cfg.index_cache):
            with open(self.cfg.index_cache, "rb") as f:
                self.items = pickle.load(f)
            if verbose:
                print(f"[DROIDPrecomp] index loaded from cache: {len(self.items)} episodes")
            return

        if self.cfg.chunks:
            chunk_dirs = [os.path.join(self.cfg.feats_root, c) for c in self.cfg.chunks]
        else:
            chunk_dirs = sorted(
                d for d in glob.glob(os.path.join(self.cfg.feats_root, "chunk-*"))
                if os.path.isdir(d)
            )

        for d in chunk_dirs:
            self.items.extend(self._scan_one(d))

        if verbose:
            print(f"[DROIDPrecomp] chunks={len(chunk_dirs)} episodes={len(self.items)}")

        if self.cfg.filter_success or self.cfg.filter_empty_L:
            self._apply_parquet_filters(verbose=verbose)

        if self.cfg.index_cache:
            os.makedirs(os.path.dirname(self.cfg.index_cache) or ".", exist_ok=True)
            with open(self.cfg.index_cache, "wb") as f:
                pickle.dump(self.items, f)
            if verbose:
                print(f"[DROIDPrecomp] index written to {self.cfg.index_cache}")

    def _apply_parquet_filters(self, verbose: bool = True) -> None:
        import pandas as pd
        if not self.cfg.droid_root:
            raise ValueError(
                "DROIDPrecompConfig.droid_root must be set when "
                "filter_success or filter_empty_L is True"
            )

        cols: list[str] = []
        if self.cfg.filter_success:
            cols.append("is_episode_successful")
        if self.cfg.filter_empty_L:
            cols.extend([
                "language_instruction",
                "language_instruction_2",
                "language_instruction_3",
            ])

        kept: list[tuple] = []
        n_drop_success = n_drop_empty = n_drop_missing = 0
        for item in self.items:
            vpath = item[0]
            chunk = os.path.basename(os.path.dirname(vpath))
            base = os.path.basename(vpath)[: -len("_vision.npy")]
            parquet = os.path.join(self.cfg.droid_root, "data", chunk, base + ".parquet")
            if not os.path.isfile(parquet):
                n_drop_missing += 1
                continue
            try:
                df = pd.read_parquet(parquet, columns=cols)
                if self.cfg.filter_success:
                    v = df["is_episode_successful"].iloc[0]
                    if v is None or not bool(v):
                        n_drop_success += 1
                        continue
                if self.cfg.filter_empty_L:
                    L1 = str(df["language_instruction"].iloc[0]).strip()
                    L2 = str(df["language_instruction_2"].iloc[0]).strip()
                    L3 = str(df["language_instruction_3"].iloc[0]).strip()
                    if L1 == "" and L2 == "" and L3 == "":
                        n_drop_empty += 1
                        continue
            except Exception:
                n_drop_missing += 1
                continue
            kept.append(item)

        n_before = len(self.items)
        self.items = kept
        if verbose:
            print(f"[DROIDPrecomp] filter: kept {len(kept)} of {n_before} "
                  f"(dropped {n_drop_success} unsuccessful, "
                  f"{n_drop_empty} empty-L, "
                  f"{n_drop_missing} missing/error)")

    # ---- access -----------------------------------------------------------

    def __len__(self) -> int:
        expansion = max(1, int(self.cfg.paraphrase_expansion))
        return expansion * len(self.items)

    def __getitem__(self, i: int) -> dict:
        mm = "r" if self.cfg.mmap else None
        expansion = max(1, int(self.cfg.paraphrase_expansion))
        ep_idx = i // expansion
        para_idx = (i % expansion) if expansion > 1 else None

        vpath, lpath, aseq_path, pseq_path = self.items[ep_idx]
        v = np.load(vpath, mmap_mode=mm)            # (1, 256, 768) fp16
        l = np.load(lpath, mmap_mode=mm)            # (3, 32, 768)  fp16
        a_seq = np.load(aseq_path, mmap_mode=mm)    # (N_ep, 14)    fp32
        p_seq = np.load(pseq_path, mmap_mode=mm)    # (N_ep, 20)    fp32

        n_ep = min(int(a_seq.shape[0]), int(p_seq.shape[0]))
        n_keep = min(n_ep, int(self.cfg.max_seq_len))

        vision_feat = torch.from_numpy(np.asarray(v, dtype=np.float32))
        lang_all = torch.from_numpy(np.asarray(l, dtype=np.float32))
        if para_idx is not None and para_idx < lang_all.shape[0]:
            lang_feat = lang_all[para_idx]
        else:
            lang_feat = lang_all.mean(dim=0)
        actions_seq = torch.from_numpy(np.asarray(a_seq[:n_keep], dtype=np.float32))
        proprio_seq = torch.from_numpy(np.asarray(p_seq[:n_keep], dtype=np.float32))

        return {
            "vision_feat":  vision_feat,
            "lang_feat":    lang_feat,
            "lang_feat_raw": lang_all,
            "actions_seq":  actions_seq,
            "proprio_seq":  proprio_seq,
            "n_ep_valid":   int(n_keep),
            "embodiment_id": self.EMBODIMENT_ID,
            "dataset_name": "droid",
        }


def collate_precomp(batch: list[dict]) -> dict:
    B = len(batch)
    max_N = max(int(b["n_ep_valid"]) for b in batch)
    actions_seq = torch.zeros(B, max_N, 14, dtype=torch.float32)
    proprio_seq = torch.zeros(B, max_N, 20, dtype=torch.float32)
    seq_valid = torch.zeros(B, max_N, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = int(b["n_ep_valid"])
        actions_seq[i, :n] = b["actions_seq"]
        proprio_seq[i, :n] = b["proprio_seq"]
        seq_valid[i, :n] = True
    return {
        "vision_feat":   torch.stack([b["vision_feat"]   for b in batch], 0),
        "lang_feat":     torch.stack([b["lang_feat"]     for b in batch], 0),
        "lang_feat_raw": torch.stack([b["lang_feat_raw"] for b in batch], 0),
        "actions_seq":   actions_seq,
        "proprio_seq":   proprio_seq,
        "seq_valid_mask": seq_valid,
        "embodiment_id": torch.tensor([b["embodiment_id"] for b in batch], dtype=torch.long),
        "dataset_name":  [b["dataset_name"] for b in batch],
    }
