"""
Precompute frozen-backbone features for one DROID chunk.

Outputs per episode under ``<out_dir>/<chunk>/``:
    episode_XXXXXX_vision.npy   float16   (1, 256, 768)   -- DINOv2 patches, frame 0
    episode_XXXXXX_lang.npy     float16   (3, n_tok, 768) -- T5 encodings, 3 paraphrases

Run one process per GPU in parallel for all chunks. Resume-safe: episodes
with both files present are skipped unless ``--force_vision`` is set.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m jepa3way.scripts.precompute_features \\
        --config jepa3way/configs/v7_11.yaml \\
        --data_root /path/to/droid \\
        --out_dir   /path/to/droid_feats \\
        --chunk     chunk-000
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from jepa3way.data.video import decode_frames


def _preprocess_clip(frames_u8: np.ndarray, img_hw: int = 224) -> torch.Tensor:
    """Resize shorter side to ``img_hw``, center-crop, scale to [0, 1], BHWC->BCHW."""
    x = torch.from_numpy(frames_u8).float() / 255.0
    x = x.permute(0, 3, 1, 2)
    _, _, H0, W0 = x.shape
    scale = img_hw / min(H0, W0)
    newH, newW = int(round(H0 * scale)), int(round(W0 * scale))
    x = F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)
    top = (newH - img_hw) // 2
    left = (newW - img_hw) // 2
    return x[:, :, top:top + img_hw, left:left + img_hw].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",       required=True)
    ap.add_argument("--data_root",    required=True)
    ap.add_argument("--out_dir",      required=True)
    ap.add_argument("--chunk",        default="chunk-000")
    ap.add_argument("--batch_eps",    type=int, default=32,
                    help="episodes per GPU batch for DINOv2/T5 inference")
    ap.add_argument("--camera",       default="observation.images.exterior_1_left")
    ap.add_argument("--force_vision", action="store_true",
                    help="reprocess episodes even if outputs already exist")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))["model"]
    n_tok = cfg["language_tokens"]
    vis_weights = cfg["vision_weights"]
    lang_weights = cfg["language_weights"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  chunk={args.chunk}  n_tok={n_tok}", flush=True)

    from transformers import Dinov2Model, T5EncoderModel, T5TokenizerFast
    print("loading DINOv2...", flush=True)
    dino = Dinov2Model.from_pretrained(vis_weights).to(device).eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    print("loading T5...", flush=True)
    t5 = T5EncoderModel.from_pretrained(lang_weights).to(device).eval()
    for p in t5.parameters():
        p.requires_grad_(False)
    tok = T5TokenizerFast.from_pretrained(lang_weights)

    parquet_dir = os.path.join(args.data_root, "data", args.chunk)
    video_dir = os.path.join(args.data_root, "videos", args.chunk, args.camera)
    out_chunk = os.path.join(args.out_dir, args.chunk)
    os.makedirs(out_chunk, exist_ok=True)

    import pandas as pd
    files = sorted(glob.glob(os.path.join(parquet_dir, "episode_*.parquet")))
    if not files:
        print(f"no parquets in {parquet_dir}")
        sys.exit(1)

    episodes = []
    for path in files:
        base = os.path.basename(path)
        ep_idx = int(base[len("episode_"):-len(".parquet")])
        vpath = os.path.join(video_dir, f"episode_{ep_idx:06d}.mp4")
        if not os.path.isfile(vpath):
            continue
        vis_out = os.path.join(out_chunk, f"episode_{ep_idx:06d}_vision.npy")
        lng_out = os.path.join(out_chunk, f"episode_{ep_idx:06d}_lang.npy")
        if (os.path.isfile(vis_out) and os.path.isfile(lng_out)
                and not args.force_vision):
            continue
        df = pd.read_parquet(path, columns=[
            "language_instruction", "language_instruction_2", "language_instruction_3",
            "frame_index",
        ])
        langs = []
        for col in ("language_instruction", "language_instruction_2", "language_instruction_3"):
            v = df[col].iloc[0]
            if isinstance(v, str) and len(v.strip()) > 0:
                langs.append(v.strip())
        if not langs:
            continue
        while len(langs) < 3:
            langs.append(langs[-1])
        episodes.append({
            "ep_idx": ep_idx, "langs": langs[:3],
            "video": vpath, "vis_out": vis_out, "lng_out": lng_out,
        })

    total = len(episodes)
    print(f"episodes to process: {total}", flush=True)
    if total == 0:
        print("nothing to do.")
        return

    n_decode_fail = 0
    B = args.batch_eps
    for batch_start in range(0, total, B):
        batch_in = episodes[batch_start: batch_start + B]

        # Decode first so per-episode failures drop cleanly.
        ok_batch = []
        ok_frames = []
        for ep in batch_in:
            try:
                raw = decode_frames(ep["video"], [0])
            except Exception as e:
                n_decode_fail += 1
                print(f"[skip] decode fail ep={ep['ep_idx']:06d}: {e}", flush=True)
                continue
            ok_batch.append(ep)
            ok_frames.append(_preprocess_clip(raw))
        if not ok_batch:
            continue

        # Vision (GPU): start frame only.
        vid_batch = torch.cat(ok_frames, dim=0).to(device)
        vid_batch = (vid_batch - mean) / std
        with torch.no_grad():
            vis_tokens = dino(pixel_values=vid_batch).last_hidden_state
        vis_tokens = vis_tokens[:, 1:, :].cpu().to(torch.float16)  # drop CLS
        vis_tokens = vis_tokens.unsqueeze(1)                        # (B, 1, 256, 768)
        for ep, vt in zip(ok_batch, vis_tokens):
            np.save(ep["vis_out"], vt.numpy())

        # Language (GPU): all 3 paraphrases per episode encoded at once.
        all_texts = [txt for ep in ok_batch for txt in ep["langs"]]
        enc = tok(all_texts, padding="max_length", truncation=True,
                  max_length=n_tok, return_tensors="pt")
        with torch.no_grad():
            lang_out = t5(input_ids=enc["input_ids"].to(device),
                          attention_mask=enc["attention_mask"].to(device)).last_hidden_state
        lang_out = lang_out.cpu().to(torch.float16)
        lang_out = lang_out.reshape(len(ok_batch), 3, n_tok, -1)
        for ep, lt in zip(ok_batch, lang_out):
            np.save(ep["lng_out"], lt.numpy())

        done = min(batch_start + B, total)
        print(f"[{done}/{total}] {args.chunk} last={ok_batch[-1]['ep_idx']:06d} "
              f"ok={len(ok_batch)}/{len(batch_in)} decode_fail_total={n_decode_fail}",
              flush=True)

    if n_decode_fail:
        print(f"[warn] {n_decode_fail} episodes skipped due to decode failures.",
              flush=True)
    print(f"done. output: {out_chunk}")


if __name__ == "__main__":
    main()
