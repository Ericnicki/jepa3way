"""
Train three-way JEPA on precomputed DROID features.

The training loop reads per-episode .npy files written by the precompute
pipeline (DINOv2 + T5 already applied offline) so the GPU never has to run
the frozen backbones at training time.

Usage (single GPU):
    python -m jepa3way.scripts.train_droid \\
        --config     jepa3way/configs/v7_11.yaml \\
        --feats_root /path/to/droid_feats \\
        --droid_root /path/to/droid \\
        --stats      /path/to/droid/meta/droid_stats.json \\
        --out_dir    /path/to/runs/v7_11

The episode index is cached to ``<out_dir>/precomp_index.pkl`` on first run
(~few seconds for 73k episodes); subsequent launches load it instantly.
``--droid_root`` is required because the dataset filters episodes by
``is_episode_successful`` and language-presence, both of which live in the
original parquets.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from jepa3way.data.droid_precomp import (
    DROIDPrecompConfig, DROIDPrecompDataset, collate_precomp,
)
from jepa3way.engine.train import main as train_main


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       required=True)
    ap.add_argument("--feats_root",   required=True,
                    help="directory containing chunk-001/..chunk-094/")
    ap.add_argument("--droid_root",   required=True,
                    help="DROID dataset root, used by the parquet filters")
    ap.add_argument("--stats",        required=True,
                    help="path to meta/droid_stats.json for the action tokenizer")
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--chunks",       default="",
                    help="comma-separated chunk names (chunk-001,chunk-002,...). "
                         "Empty = all chunks present under feats_root.")
    ap.add_argument("--max_steps",    type=int, default=None)
    ap.add_argument("--batch",        type=int, default=None,
                    help="override cfg.train.batch_per_gpu")
    ap.add_argument("--num_workers",  type=int, default=4)
    ap.add_argument("--probe_size",   type=int, default=128,
                    help="number of episodes held out for the probe loader")
    ap.add_argument("--seed",         type=int, default=0)
    ap.add_argument("--save_every",   type=int, default=20000,
                    help="save a checkpoint every N steps (0 = no periodic saves)")
    ap.add_argument("--save_milestones", default="",
                    help="comma-separated extra step numbers to save at "
                         "(e.g. 5000,15000,19000). The final step is always saved.")
    return ap.parse_args()


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _seed_everything(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.out_dir, "train.log")),
        ],
    )
    log = logging.getLogger("jepa3way")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.max_steps is not None:
        cfg["train"]["total_steps"] = args.max_steps
    if args.batch is not None:
        cfg["train"]["batch_per_gpu"] = args.batch

    if not os.path.isfile(args.stats):
        raise FileNotFoundError(
            f"missing {args.stats}. Run scripts/compute_droid_stats.py first."
        )
    os.environ["DROID_STATS_JSON"] = args.stats
    log.info(f"DROID_STATS_JSON = {args.stats}")

    chunks = [c.strip() for c in args.chunks.split(",") if c.strip()]
    data_cfg = cfg.get("data", {})
    ds_cfg = DROIDPrecompConfig(
        feats_root=args.feats_root,
        chunks=chunks,
        mmap=True,
        index_cache=os.path.join(args.out_dir, "precomp_index.pkl"),
        max_seq_len=int(data_cfg.get("max_seq_len", 64)),
        droid_root=args.droid_root,
        filter_success=bool(data_cfg.get("filter_success", False)),
        filter_empty_L=bool(data_cfg.get("filter_empty_L", False)),
        paraphrase_expansion=int(data_cfg.get("paraphrase_expansion", 1)),
    )
    ds = DROIDPrecompDataset(ds_cfg)
    n = len(ds)
    log.info(f"DROIDPrecomp dataset: {n} samples")
    if n == 0:
        raise RuntimeError(f"no precomputed episodes found under {args.feats_root}")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).tolist()
    probe_ids = perm[: args.probe_size]
    train_ids = perm[args.probe_size:]
    train_ds = Subset(ds, train_ids)
    probe_ds = Subset(ds, probe_ids)

    batch = cfg["train"]["batch_per_gpu"]
    # persistent_workers keeps the workers alive across StopIteration at
    # epoch boundaries; without it, every ~415 steps (batch=512) the workers
    # die, respawn cold, and re-fill the prefetch queue from disk.
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_precomp,
        pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    probe_loader = DataLoader(
        probe_ds, batch_size=min(batch, args.probe_size), shuffle=False,
        num_workers=1, collate_fn=collate_precomp,
        pin_memory=True, drop_last=False, persistent_workers=True,
    )
    log.info(
        f"train samples={len(train_ds)} probe samples={len(probe_ds)} "
        f"batch={batch} workers={args.num_workers} "
        f"steps/epoch~{len(train_ds)//max(batch,1)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device} bf16={cfg['train'].get('bf16', False)}")

    milestones = [int(s) for s in args.save_milestones.split(",") if s.strip()]
    train_main(
        cfg, train_loader, probe_loader, device,
        max_steps=cfg["train"]["total_steps"],
        out_dir=args.out_dir,
        save_every=args.save_every,
        save_milestones=milestones,
    )
    log.info("training done.")


if __name__ == "__main__":
    main()
