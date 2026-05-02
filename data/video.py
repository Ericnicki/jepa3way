"""
Multi-backend video frame decoder used by the precompute scripts.

Backend precedence: ``JEPA3WAY_VIDEO_BACKEND`` env override > pyav > decord >
torchvision.io. PyAV is preferred because many DROID mp4s use codecs that
prebuilt decord wheels do not support.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

_VIDEO_BACKEND: Optional[str] = None


def _pick_backend() -> str:
    global _VIDEO_BACKEND
    if _VIDEO_BACKEND is not None:
        return _VIDEO_BACKEND
    override = os.environ.get("JEPA3WAY_VIDEO_BACKEND")
    if override:
        import importlib
        modname = {"av": "av", "decord": "decord", "tv": "torchvision.io"}.get(override)
        if modname is None:
            raise RuntimeError(f"unknown JEPA3WAY_VIDEO_BACKEND={override!r}")
        importlib.import_module(modname)
        _VIDEO_BACKEND = override
        return _VIDEO_BACKEND
    for name, modname in (("av", "av"), ("decord", "decord"), ("tv", "torchvision.io")):
        try:
            __import__(modname)
            _VIDEO_BACKEND = name
            return _VIDEO_BACKEND
        except Exception:
            continue
    raise RuntimeError(
        "No video decoder available. Install one of: av (pyav), decord, "
        "torchvision with video support."
    )


def decode_frames(path: str, frame_idx: list[int]) -> np.ndarray:
    """Return ``(K, H, W, 3)`` uint8. Out-of-range indices are clamped."""
    backend = _pick_backend()
    idx = np.asarray(frame_idx, dtype=np.int64)

    if backend == "decord":
        from decord import VideoReader, cpu
        vr = VideoReader(path, ctx=cpu(0))
        idx_clamped = np.clip(idx, 0, len(vr) - 1)
        return vr.get_batch(idx_clamped.tolist()).asnumpy()

    if backend == "av":
        import av
        container = av.open(path)
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            total = stream.frames or 0
            idx_clamped = np.clip(idx, 0, max(total - 1, 0)) if total else idx
            wanted = set(int(x) for x in idx_clamped)
            out = {}
            for i, frame in enumerate(container.decode(video=0)):
                if i in wanted:
                    out[i] = frame.to_ndarray(format="rgb24")
                if len(out) == len(wanted):
                    break
        finally:
            container.close()
        if not out:
            raise RuntimeError(f"av decoded 0 frames from {path}")
        last = max(out.keys())
        return np.stack([out.get(int(i), out[last]) for i in idx_clamped])

    if backend == "tv":
        import torchvision.io as tvio
        vid, _, _ = tvio.read_video(path, pts_unit="sec", output_format="THWC")
        idx_clamped = np.clip(idx, 0, vid.shape[0] - 1)
        return vid[idx_clamped].numpy()

    raise RuntimeError(f"unknown backend {backend}")
