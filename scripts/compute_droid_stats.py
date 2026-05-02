"""
Compute per-slot mean/std for DROID actions, mapped into the 14-slot canonical
schema, and write them to ``<root>/meta/droid_stats.json`` so that
``data/action_tokenizer.py`` can load real numbers.

Canonical mapping (Franka 7-DoF + gripper):
    slots 0..5  <- action.cartesian_velocity   (6)
    slot  6     <- action.gripper_position     (1)
    slots 7..13 <- action.joint_velocity       (7)

Only counts steps from episodes that have at least one non-empty language
instruction. Without this filter, episodes with no language pollute the
distribution and the empty-L episodes would also need to be excluded from
training (which they are, separately, via ``filter_empty_L``).

Usage:
    python -m jepa3way.scripts.compute_droid_stats --root /path/to/droid
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd


def load_canonical_actions(df: pd.DataFrame) -> np.ndarray:
    cart = np.stack([np.asarray(x, dtype=np.float32)
                     for x in df["action.cartesian_velocity"]])
    grip = np.asarray(df["action.gripper_position"].to_list(),
                      dtype=np.float32).reshape(-1, 1)
    joint = np.stack([np.asarray(x, dtype=np.float32)
                      for x in df["action.joint_velocity"]])
    return np.concatenate([cart, grip, joint], axis=1).astype(np.float32)


def episode_has_language(df: pd.DataFrame) -> bool:
    for col in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        v = df[col].iloc[0]
        if isinstance(v, str) and len(v.strip()) > 0:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="DROID dataset root (containing data/ and meta/)")
    ap.add_argument("--chunk", default="chunk-000")
    args = ap.parse_args()

    pattern = os.path.join(args.root, "data", args.chunk, "episode_*.parquet")
    files = sorted(glob.glob(pattern))
    print(f"scanning {len(files)} parquets matching {pattern}")

    # Welford online mean/variance.
    count = 0
    mean = np.zeros(14, dtype=np.float64)
    M2 = np.zeros(14, dtype=np.float64)
    n_skip_empty_lang = 0
    n_episodes_used = 0

    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        if not episode_has_language(df):
            n_skip_empty_lang += 1
            continue
        n_episodes_used += 1
        x = load_canonical_actions(df)
        for row in x:
            count += 1
            delta = row - mean
            mean += delta / count
            delta2 = row - mean
            M2 += delta * delta2
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} parquets, {n_episodes_used} eps used, steps={count}")

    var = M2 / max(count - 1, 1)
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    mean = mean.astype(np.float32)

    stats = {
        "name": "droid",
        "slots": ["ee_dx", "ee_dy", "ee_dz", "ee_rx", "ee_ry", "ee_rz",
                  "gripper", "j1", "j2", "j3", "j4", "j5", "j6", "j7"],
        "source_columns": {
            "slots_0_5":  "action.cartesian_velocity",
            "slot_6":     "action.gripper_position",
            "slots_7_13": "action.joint_velocity",
        },
        "mu":    mean.tolist(),
        "sigma": std.tolist(),
        "mask":  [1.0] * 14,
        "n_steps": int(count),
        "n_episodes_used": int(n_episodes_used),
        "n_episodes_skipped_empty_lang": int(n_skip_empty_lang),
    }
    out = os.path.join(args.root, "meta", "droid_stats.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fo:
        json.dump(stats, fo, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
