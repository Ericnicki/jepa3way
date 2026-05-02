"""
Precompute per-episode action or proprio sequences at FIXED wall-clock stride.

Sampling rule: every episode is sampled at frames ``np.arange(0, ep_length,
STRIDE)``. DROID is 15 Hz native, so STRIDE=3 yields 5 Hz effective.
``N_ep = ceil(ep_length / STRIDE)`` is variable across episodes; the wall-
clock interval between samples is constant (~0.2 s).

Outputs (per episode under ``<feats_root>/<chunk>/``):
    --mode actions  ->  episode_XXXXXX_actions_seq.npy   float32 (N_ep, 14)
    --mode proprio  ->  episode_XXXXXX_proprio_seq.npy   float32 (N_ep, 20)

Action slot layout (matches compute_droid_stats.py):
    slots 0..5  <- action.cartesian_velocity   (6)
    slot  6     <- action.gripper_position     (1)
    slots 7..13 <- action.joint_velocity       (7)

Proprio slot layout:
    slots 0..5   <- observation.state.cartesian_position    (6)
    slot  6      <- observation.state.gripper_position      (1)
    slots 7..13  <- observation.state.joint_position        (7)
    slots 14..19 <- camera_extrinsics.exterior_1_left       (6)

Resume-safe: episodes with a sane existing output are skipped; pass
``--force`` to overwrite. Episodes with ``N_ep < min_n_ep`` are skipped
entirely. The two modes MUST be run with matching ``--stride``,
``--min_n_ep``, and ``--max_n_ep`` so the per-episode lengths line up.

Usage:
    python -m jepa3way.scripts.precompute_seq --mode actions \\
        --droid_root /path/to/droid \\
        --feats_root /path/to/droid_feats \\
        --num_workers 16

    python -m jepa3way.scripts.precompute_seq --mode proprio \\
        --droid_root /path/to/droid \\
        --feats_root /path/to/droid_feats \\
        --num_workers 16
"""
from __future__ import annotations

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np


STRIDE = 3
MIN_N_EP = 8


# Per-mode wiring: (output_dim, output_suffix, parquet columns, slot dims).
_MODES = {
    "actions": dict(
        out_dim=14,
        suffix="_actions_seq.npy",
        cols=[
            "action.cartesian_velocity",
            "action.gripper_position",
            "action.joint_velocity",
        ],
        slot_dims=(6, 1, 7),
    ),
    "proprio": dict(
        out_dim=20,
        suffix="_proprio_seq.npy",
        cols=[
            "observation.state.cartesian_position",
            "observation.state.gripper_position",
            "observation.state.joint_position",
            "camera_extrinsics.exterior_1_left",
        ],
        slot_dims=(6, 1, 7, 6),
    ),
}


def _to_vec(x, expect_dim: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size != expect_dim:
        raise ValueError(f"expected dim={expect_dim}, got shape {arr.shape}")
    return arr


def _frame_indices(ep_length: int, stride: int) -> np.ndarray:
    return np.arange(0, ep_length, stride, dtype=np.int64)


def _read_row(df, cols: list[str], slot_dims: tuple[int, ...], i: int) -> np.ndarray:
    parts = []
    for col, dim in zip(cols, slot_dims):
        if dim == 1:
            val = df[col].iloc[i]
            try:
                parts.append(_to_vec(val, 1))
            except ValueError:
                parts.append(np.asarray([float(val)], dtype=np.float32))
        else:
            parts.append(_to_vec(df[col].iloc[i], dim))
    return np.concatenate(parts, axis=0)


def _extract_one(args_tuple):
    parquet_path, out_path, force, stride, min_n_ep, max_n_ep, mode = args_tuple
    spec = _MODES[mode]
    out_dim = spec["out_dim"]

    if not force and os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        try:
            existing = np.load(out_path, mmap_mode="r")
            if (existing.ndim == 2
                    and existing.shape[1] == out_dim
                    and existing.shape[0] >= min_n_ep
                    and (max_n_ep <= 0 or existing.shape[0] <= max_n_ep)
                    and existing.dtype == np.float32):
                return ("skip", parquet_path, None)
        except Exception:
            pass

    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path, columns=spec["cols"])
        L = len(df)
        if L < 1:
            return ("err", parquet_path, "empty parquet")

        idx = _frame_indices(L, stride)
        n_ep = idx.size
        if n_ep < min_n_ep:
            return ("short", parquet_path, f"N_ep={n_ep} < {min_n_ep}")
        if max_n_ep > 0 and n_ep > max_n_ep:
            if os.path.isfile(out_path):
                try:
                    os.remove(out_path)
                except OSError:
                    pass
            return ("too_long", parquet_path, f"N_ep={n_ep} > {max_n_ep}")

        out = np.zeros((n_ep, out_dim), dtype=np.float32)
        for k, i in enumerate(idx):
            out[k] = _read_row(df, spec["cols"], spec["slot_dims"], int(i))
        np.save(out_path, out)
        return ("ok", parquet_path, None)
    except Exception as e:
        return ("err", parquet_path, f"{type(e).__name__}: {e}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",        required=True, choices=list(_MODES))
    ap.add_argument("--droid_root",  required=True,
                    help="path containing data/chunk-*/episode_*.parquet")
    ap.add_argument("--feats_root",  required=True,
                    help="path containing chunk-*/ for output .npy files")
    ap.add_argument("--chunks",      default="",
                    help="comma-separated chunk IDs (e.g. '001,002'); empty = all chunks")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--stride",      type=int, default=STRIDE)
    ap.add_argument("--min_n_ep",    type=int, default=MIN_N_EP)
    ap.add_argument("--max_n_ep",    type=int, default=0,
                    help="if >0, drop episodes with N_ep > max_n_ep (no file written)")
    ap.add_argument("--force",       action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    chunk_filter = set(c.strip() for c in args.chunks.split(",") if c.strip())
    suffix = _MODES[args.mode]["suffix"]

    tasks: list[tuple] = []
    chunk_dirs = sorted(glob.glob(os.path.join(args.droid_root, "data", "chunk-*")))
    for chunk_dir in chunk_dirs:
        chunk_name = os.path.basename(chunk_dir)
        chunk_id = chunk_name.replace("chunk-", "")
        if chunk_filter and chunk_id not in chunk_filter:
            continue
        feats_chunk_dir = os.path.join(args.feats_root, chunk_name)
        os.makedirs(feats_chunk_dir, exist_ok=True)
        for parquet in sorted(glob.glob(os.path.join(chunk_dir, "episode_*.parquet"))):
            base = os.path.basename(parquet)[: -len(".parquet")]
            out_path = os.path.join(feats_chunk_dir, base + suffix)
            tasks.append((parquet, out_path, args.force,
                          args.stride, args.min_n_ep, args.max_n_ep, args.mode))

    print(f"[main] mode={args.mode} stride={args.stride} min_n_ep={args.min_n_ep} "
          f"{len(tasks)} tasks across "
          f"{len(set(os.path.basename(os.path.dirname(t[0])) for t in tasks))} chunks",
          flush=True)
    if not tasks:
        return

    counts = {"ok": 0, "skip": 0, "short": 0, "too_long": 0, "err": 0}
    samples: dict[str, list[str]] = {"short": [], "too_long": [], "err": []}
    sample_cap = {"short": 5, "too_long": 5, "err": 10}

    with Pool(args.num_workers) as pool:
        for i, (status, path, msg) in enumerate(
                pool.imap_unordered(_extract_one, tasks, chunksize=8)):
            counts[status] = counts.get(status, 0) + 1
            if status in samples and len(samples[status]) < sample_cap[status]:
                samples[status].append(f"{path}: {msg}")
            if (i + 1) % 2000 == 0 or (i + 1) == len(tasks):
                print(f"[{i+1}/{len(tasks)}] " + " ".join(
                    f"{k}={counts[k]}" for k in ("ok", "skip", "short", "too_long", "err")
                ), flush=True)

    print(f"[main] DONE: " + " ".join(
        f"{k}={counts[k]}" for k in ("ok", "skip", "short", "too_long", "err")
    ) + f" total={len(tasks)}", flush=True)
    for kind in ("short", "too_long", "err"):
        if samples[kind]:
            print(f"[main] first {kind} samples:", flush=True)
            for s in samples[kind]:
                print(f"  {s}", flush=True)


if __name__ == "__main__":
    main()
