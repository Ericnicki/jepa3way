# jepa3way

A three-way Joint-Embedding Predictive Architecture for vision-language-action
representation learning. One transformer fuser is invoked three times per
training step over modality-pair views of the input — `(L, V)`, `(V, A)`, and
`(L, A)` — and trained to produce a single shared embedding `S` that is
invariant to which view it sees. Trained on DROID with a BYOL-style
leave-one-out consensus target plus VICReg.

## Architecture

```
       L (T5)         V (DINOv2)         A (action + proprio)
        │                 │                       │
       proj_L          proj_V + LN             act_embed
        │                 │                       │
        └──────┐  ┌───────┴───────┐  ┌────────────┘
              [SHARED_Q | L | V | A]   ← one of A/V/L is content-zeroed
                       │                  per branch; type embeddings tag
                  TriadicFuser            every slot so the fuser can
                       │                  address even the zeroed slot
                z ∈ ℝ^512  ──► VICReg
                       │
                  Projector g            ──┐
                       │                   │  align_loss_loo against
                      zp                   │  L2(mean of the OTHER TWO
                       │                   │  branches' EMA projector outputs)
                  Predictor h           ──┘
```

Per-branch zeroing pattern:

| Branch | L     | V     | A     |
|--------|-------|-------|-------|
| LV     | real  | real  | zero  |
| VA     | zero  | real  | real  |
| LA     | real  | zero  | real  |

The fuser, projector, and predictor have **shared weights** across all three
branches. The branch identity is encoded entirely by zeroing and by the
4-class type embedding (query / L / V / A) that tags every slot.

## Repository layout

```
jepa3way/
├── configs/
│   └── v7_11.yaml                # the single config
├── data/
│   ├── action_tokenizer.py       # 14-slot canonical action schema + registry
│   ├── droid_precomp.py          # precomputed-feature dataset + collate
│   └── video.py                  # multi-backend mp4 frame decoder
├── engine/
│   ├── probes.py                 # diagnostic cosine-drift probes
│   ├── schedules.py              # EMA tau and LR cosine schedules
│   └── train.py                  # ThreeWayJEPA + train loop + checkpointing
├── losses/
│   ├── align.py                  # leave-one-out consensus alignment
│   └── vicreg.py                 # variance + covariance hinge
├── models/
│   ├── backbones.py              # frozen T5 / DINOv2 + ActionTokenEmbed
│   ├── ema.py                    # EMA wrapper + safe-forward context
│   ├── fusers.py                 # TriadicFuser (transformer trunk)
│   └── heads.py                  # BYOL projector and predictor
└── scripts/
    ├── compute_droid_stats.py    # writes meta/droid_stats.json
    ├── precompute_features.py    # DINOv2 + T5 -> per-episode .npy
    ├── precompute_seq.py         # actions and proprio at fixed wall-clock stride
    └── train_droid.py            # training entry point
```

## Setup

Requirements: PyTorch ≥ 2.1, transformers, pandas, numpy, pyyaml, and one
of `pyav` / `decord` / `torchvision` (for video decoding at precompute
time only). Install the frozen backbones locally:

* `t5-base` — https://huggingface.co/google-t5/t5-base
* `dinov2-base` — https://huggingface.co/facebook/dinov2-base

Then update `model.language_weights` and `model.vision_weights` in
[`configs/v7_11.yaml`](configs/v7_11.yaml) to point at your local copies.

## Pipeline

The full pipeline is offline-precompute → train. The trainer never touches
mp4 files or runs the frozen backbones.

### 1. Action statistics

```bash
python -m jepa3way.scripts.compute_droid_stats --root /path/to/droid
```

Writes `<root>/meta/droid_stats.json` with per-slot mu / sigma / mask. Only
counts steps from episodes that have at least one non-empty language
instruction (so the stats and the training set agree about which episodes
matter).

### 2. Frozen-backbone features

Run one process per GPU in parallel for all chunks:

```bash
CUDA_VISIBLE_DEVICES=0 python -m jepa3way.scripts.precompute_features \
    --config jepa3way/configs/v7_11.yaml \
    --data_root /path/to/droid \
    --out_dir   /path/to/droid_feats \
    --chunk     chunk-000
```

Per episode this writes:

* `episode_XXXXXX_vision.npy` — `(1, 256, 768)` fp16, DINOv2 patches at frame 0
* `episode_XXXXXX_lang.npy`   — `(3, 32, 768)` fp16, T5 encodings of all 3 paraphrases

### 3. Action and proprio sequences

Both modes must be run with the same `--stride`, `--min_n_ep`, and
`--max_n_ep` so per-episode lengths line up:

```bash
python -m jepa3way.scripts.precompute_seq --mode actions \
    --droid_root /path/to/droid \
    --feats_root /path/to/droid_feats \
    --num_workers 16

python -m jepa3way.scripts.precompute_seq --mode proprio \
    --droid_root /path/to/droid \
    --feats_root /path/to/droid_feats \
    --num_workers 16
```

Default `--stride 3` yields 5 Hz at DROID's native 15 Hz. Each output is
`(N_ep, 14)` (actions) or `(N_ep, 20)` (proprio), with `N_ep` variable
across episodes.

### 4. Train

```bash
CUDA_VISIBLE_DEVICES=0 python -m jepa3way.scripts.train_droid \
    --config     jepa3way/configs/v7_11.yaml \
    --feats_root /path/to/droid_feats \
    --droid_root /path/to/droid \
    --stats      /path/to/droid/meta/droid_stats.json \
    --out_dir    /path/to/runs/v7_11
```

Outputs:

* `<out_dir>/train.log`   — per-step loss + every-N-step probe metrics
* `<out_dir>/precomp_index.pkl` — cached episode index (skip rescan on resume)
* `<out_dir>/ckpts/`      — `step_*.pt`, `final_*.pt`, plus `interrupt_*.pt`
                            on Ctrl-C / SIGTERM

`SIGTERM`/`SIGINT` save current state before exit. Re-launching the same
command auto-resumes from the latest checkpoint.

## Configuration

Everything is set in [`configs/v7_11.yaml`](configs/v7_11.yaml). The keys
that matter most:

| Key | What it does |
|-----|---|
| `train.batch_per_gpu` | Per-GPU batch size. 512 fits a single A800 in bf16. |
| `train.total_steps` | Length of the run; the LR cosine and tau schedules decay over this window. |
| `train.probe_every` | Frequency of the diagnostic probe pass. 0 disables. |
| `loss.alpha_align`, `loss.alpha_vic` | Loss weights. Defaults reflect the trained run. |
| `data.max_seq_len` | Per-episode action/proprio cap. Episodes longer than this are truncated. |
| `data.paraphrase_expansion` | Materialize each episode as N samples, one per paraphrase. 1 = mean-pool. |
| `data.filter_success`, `data.filter_empty_L` | Index-time quality filters; require `--droid_root`. |

## Diagnostic probes

Probes are computed on a fixed held-out batch every `train.probe_every`
steps. They are forward-only and never abort the run. Key metrics:

* **`probe_p3`** — cosine drift on `z_lv` when L is shuffled across the
  batch. Higher = more language-sensitive.
* **`probe_p4`** — per-sample consensus (cosine sim averaged over branch
  pairs). High is the goal: it means the three branches produce the same
  shared `S` per sample.
* **`probe_p5_ratio`** — `p3 / p5_null`. Close to 1.0 means swapping
  languages is no different than removing them; well above 1.0 indicates
  semantic grounding.
* **`probe_lswap_la / vswap_lv / vswap_va / aswap_la / aswap_va / pswap_*`**
  — per-branch sensitivity to swapping each modality. Tells you which
  modality each branch is actually using.
* **`probe_cos_lv_va / lv_la / va_la`** — pairwise inter-branch cosine
  similarity. A direct proxy for how close the branches already agree.
