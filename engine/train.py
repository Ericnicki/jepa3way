"""
Three-way JEPA training loop.

Architecture
------------
One TriadicFuser is invoked three times per step, once per branch. Each
branch sees the canonical [SHARED_Q | L | V | A] token layout, with the
modality not in the branch name replaced by content-zero tokens that still
carry their type embedding so the fuser can address the slot:

    LV branch : L=real,  V=real,  A=zero
    VA branch : L=zero,  V=real,  A=real
    LA branch : L=real,  V=zero,  A=real

The fuser reads out the SHARED_Q slot, projects to ``d_shared``, and feeds
into the BYOL projector ``g`` and predictor ``h``. The training target is
the leave-one-out consensus of the EMA projector outputs, so each online
branch is asked to predict the average of the *other two* branches' EMA
representations.

Loss = alpha_align * align_loss_loo + alpha_vic * vicreg_loss(z)

Invariants
----------
* VICReg is applied on the pre-projector fuser output ``z`` (one call per
  branch), never on the projector output ``zp`` -- regularizing zp eats
  modality bandwidth in the projector and silences the A path.
* The EMA target path always runs under ``ema_forward`` (eval + no_grad).
* The probe loader yields the same first batch every time so probe
  trajectories are comparable across steps.
"""
from __future__ import annotations

import logging
import os
import signal
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.action_tokenizer import default_registry, tokenize_batch
from ..losses.align import align_loss_loo, consensus_target_loo
from ..losses.vicreg import vicreg_loss
from ..models.backbones import ActionTokenEmbed
from ..models.ema import EMAWrapper, ema_forward
from ..models.fusers import TriadicFuser
from ..models.heads import Predictor, Projector
from . import probes
from .schedules import cosine_tau, lr_with_warmup_cosine

logger = logging.getLogger("jepa3way")


# ── model ─────────────────────────────────────────────────────────────────────

class ThreeWayJEPA(nn.Module):
    """Online (student) branch of the three-way JEPA.

    Inputs are precomputed DINOv2 / T5 features, so neither vision nor
    language backbones live inside this module -- only the trainable
    projections, action embedder, fuser, and BYOL heads.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.d_model = m["d_model"]
        self.d_shared = m["d_shared"]
        self.proprio_dim = int(m["proprio_dim"])

        self.proj_L = nn.Linear(m["language_dim"], m["d_model"])
        self.proj_V = nn.Linear(m["video_dim"], m["d_model"])
        # LayerNorm on proj_V output equalizes per-token magnitudes across
        # modalities. Without it, frozen DINOv2 patch tokens come out roughly
        # 100x larger than T5 / action tokens and the fuser's attention
        # softmax saturates on V by magnitude alone.
        self.proj_V_ln = nn.LayerNorm(m["d_model"])

        self.act_embed = ActionTokenEmbed(
            action_slots=m["action_slots"],
            d_model=m["d_model"],
            proprio_dim=self.proprio_dim,
        )

        # Learned token used by the null-language probe (and as the target of
        # symmetric language dropout, when enabled). Shape (1, 1, d_model);
        # broadcast-expanded to (B, n_tok, d_model) at use site.
        self.null_L = nn.Parameter(torch.zeros(1, 1, m["d_model"]))
        nn.init.trunc_normal_(self.null_L, std=0.02)

        # 4-class type embedding (query / L / V / A). Index 0 is unused at
        # forward time (the fuser owns the SHARED_Q parameter); kept at width
        # 4 for state-dict compatibility with checkpoints that included it.
        self.type_emb = nn.Embedding(4, m["d_model"])
        nn.init.trunc_normal_(self.type_emb.weight, std=0.02)

        self.fuser = TriadicFuser(
            d_model=m["d_model"], d_shared=m["d_shared"],
            n_heads=m["n_heads"], ffn=m["ffn"],
            n_layers=m["fuser_layers"], dropout=m["dropout"],
        )
        self.projector = Projector(m["d_shared"], m["proj_hidden"], m["proj_out"])
        self.predictor = Predictor(m["proj_out"], m["proj_hidden"], m["proj_out"])

    # ---- per-modality encoders --------------------------------------------

    def encode_L(self, lang_feat: torch.Tensor,
                 lang_drop_mask: torch.Tensor) -> torch.Tensor:
        """Project precomputed T5 features. Replace samples flagged by
        ``lang_drop_mask`` with the learned null-L token; used by the
        null-language probe."""
        l = self.proj_L(lang_feat)
        if lang_drop_mask.any():
            B, N, D = l.shape
            null = self.null_L.expand(B, N, D)
            mm = lang_drop_mask.view(B, 1, 1).to(l.dtype)
            l = l * (1.0 - mm) + null * mm
        return l

    def encode_V(self, v_raw: torch.Tensor) -> torch.Tensor:
        """Project precomputed DINOv2 features. v_raw: (B, T_v, N_v, D_in)."""
        B, T, N, D_in = v_raw.shape
        v = self.proj_V(v_raw.reshape(B, T * N, D_in))
        return self.proj_V_ln(v)

    def encode_A(self, action_feats: torch.Tensor,
                 proprio_seq: torch.Tensor) -> torch.Tensor:
        return self.act_embed(action_feats, proprio_seq)

    # ---- branch dispatch --------------------------------------------------

    def _branches(self, l: torch.Tensor, v: torch.Tensor, a: torch.Tensor,
                  action_kpm: torch.Tensor, fuser_module) -> tuple:
        """Run all three branches through ``fuser_module`` and return
        ``(z_lv, z_va, z_la)``.

        Per-branch zeroing tags every slot with its type embedding even when
        zeroed out, so the fuser can still address the slot.

        action_kpm: (B, Na) bool, True at padding positions in the action
        sequence. Plumbed into branches where A is real (VA, LA). The LV
        branch's A is wholly zeroed so all Na positions carry the same
        constant content; we leave them unmasked there.

        Caveat: because Na is variable per batch (collated to per-batch max),
        the LV branch sees a variable count of identical type-embedding
        tokens and the fuser could in principle learn to count them. In
        practice this is benign -- the count carries no per-sample signal,
        only a per-batch one -- but it is the reason this branch is not
        symmetrically masked.
        """
        B, Nl, _ = l.shape
        Nv = v.shape[1]
        Na = a.shape[1]
        device = l.device

        e_l = self.type_emb.weight[1].view(1, 1, -1)
        e_v = self.type_emb.weight[2].view(1, 1, -1)
        e_a = self.type_emb.weight[3].view(1, 1, -1)

        l_real = l + e_l
        v_real = v + e_v
        a_real = a + e_a
        l_zero = e_l.expand(B, Nl, -1)
        v_zero = e_v.expand(B, Nv, -1)
        a_zero = e_a.expand(B, Na, -1)

        if action_kpm is not None:
            F_q = torch.zeros(B, 1, dtype=torch.bool, device=device)
            F_l = torch.zeros(B, Nl, dtype=torch.bool, device=device)
            F_v = torch.zeros(B, Nv, dtype=torch.bool, device=device)
            kpm_va = torch.cat([F_q, F_l, F_v, action_kpm], dim=1)
            kpm_la = torch.cat([F_q, F_l, F_v, action_kpm], dim=1)
        else:
            kpm_va = kpm_la = None

        z_lv = fuser_module(l_real, v_real, a_zero, key_padding_mask=None)
        z_va = fuser_module(l_zero, v_real, a_real, key_padding_mask=kpm_va)
        z_la = fuser_module(l_real, v_zero, a_real, key_padding_mask=kpm_la)
        return z_lv, z_va, z_la

    def forward_online(self, l: torch.Tensor, v: torch.Tensor, a: torch.Tensor,
                       action_kpm: torch.Tensor) -> dict:
        z_lv, z_va, z_la = self._branches(l, v, a, action_kpm, self.fuser)
        zp_lv, zp_va, zp_la = self.projector(z_lv), self.projector(z_va), self.projector(z_la)
        p_lv, p_va, p_la = self.predictor(zp_lv), self.predictor(zp_va), self.predictor(zp_la)
        return dict(
            z_lv=z_lv, z_va=z_va, z_la=z_la,
            zp_lv=zp_lv, zp_va=zp_va, zp_la=zp_la,
            p_lv=p_lv, p_va=p_va, p_la=p_la,
        )


# ── EMA bundle ───────────────────────────────────────────────────────────────

def build_ema_bundle(online: ThreeWayJEPA) -> nn.ModuleDict:
    """Only fuser and projector are EMA'd. The predictor is online-only
    (BYOL convention), and per-modality projections / embeddings live on the
    student side -- the EMA inputs are the *detached student outputs*."""
    return nn.ModuleDict({
        "fuser": EMAWrapper(online.fuser),
        "projector": EMAWrapper(online.projector),
    })


def ema_update_all(ema: nn.ModuleDict, online: ThreeWayJEPA, tau: float) -> None:
    ema["fuser"].update(online.fuser, tau)
    ema["projector"].update(online.projector, tau)


# ── single training step ─────────────────────────────────────────────────────

def train_step(model: ThreeWayJEPA, ema: nn.ModuleDict, batch: dict,
               registry, cfg: dict, device) -> dict:
    lcfg = cfg["loss"]

    lang_feat = batch["lang_feat"].to(device)
    v_raw = batch["vision_feat"].to(device)
    actions = batch["actions_seq"].to(device)
    proprio = batch["proprio_seq"].to(device)
    seq_valid = batch["seq_valid_mask"].to(device)
    dataset_names = batch["dataset_name"]

    B = v_raw.shape[0]
    no_drop = torch.zeros(B, dtype=torch.bool, device=device)
    action_kpm = ~seq_valid

    act_feats, _ = tokenize_batch(actions, dataset_names, registry)

    # Online (student) forward.
    l = model.encode_L(lang_feat, no_drop)
    v = model.encode_V(v_raw)
    a = model.encode_A(act_feats, proprio)
    out = model.forward_online(l, v, a, action_kpm)

    # EMA (teacher) forward. Inputs are the detached student outputs so
    # gradient flow is L_align -> student predictor only.
    with ema_forward(ema["fuser"]) as fuser_t, \
         ema_forward(ema["projector"]) as proj_t:
        z_lv_t, z_va_t, z_la_t = model._branches(
            l.detach(), v.detach(), a.detach(), action_kpm, fuser_t,
        )
        p_lv_t = proj_t(z_lv_t)
        p_va_t = proj_t(z_va_t)
        p_la_t = proj_t(z_la_t)

    t_lv, t_va, t_la = consensus_target_loo(p_lv_t, p_va_t, p_la_t)
    L_align = align_loss_loo(out["p_lv"], out["p_va"], out["p_la"], t_lv, t_va, t_la)

    L_vic = (
        vicreg_loss(out["z_lv"], lcfg["vicreg_var_coef"], lcfg["vicreg_cov_coef"], lcfg["vicreg_eps"])
        + vicreg_loss(out["z_va"], lcfg["vicreg_var_coef"], lcfg["vicreg_cov_coef"], lcfg["vicreg_eps"])
        + vicreg_loss(out["z_la"], lcfg["vicreg_var_coef"], lcfg["vicreg_cov_coef"], lcfg["vicreg_eps"])
    ) / 3.0

    total = lcfg["alpha_align"] * L_align + lcfg["alpha_vic"] * L_vic
    return {
        "total": total,
        "L_align": L_align.detach().float().item(),
        "L_vic":   L_vic.detach().float().item(),
    }


# ── probes ───────────────────────────────────────────────────────────────────

def run_probes(model: ThreeWayJEPA, batch: dict, registry, device) -> dict:
    model.eval()
    with torch.no_grad():
        lang_feat = batch["lang_feat"].to(device)
        v_raw = batch["vision_feat"].to(device)
        actions = batch["actions_seq"].to(device)
        proprio = batch["proprio_seq"].to(device)
        seq_valid = batch["seq_valid_mask"].to(device)
        dataset_names = batch["dataset_name"]

        B = v_raw.shape[0]
        no_drop = torch.zeros(B, dtype=torch.bool, device=device)
        action_kpm = ~seq_valid

        act_feats, _ = tokenize_batch(actions, dataset_names, registry)
        l = model.encode_L(lang_feat, no_drop)
        v = model.encode_V(v_raw)
        a = model.encode_A(act_feats, proprio)

        def _zs(l_in, v_in, a_in, akpm_in):
            return model._branches(l_in, v_in, a_in, akpm_in, model.fuser)

        z_lv, z_va, z_la = _zs(l, v, a, action_kpm)

        p1 = [probes.intra_diversity(z) for z in (z_lv, z_va, z_la)]
        p2 = probes.cross_mean_distinctness([z_lv, z_va, z_la])
        p4 = probes.per_sample_consensus([z_lv, z_va, z_la])

        perm = torch.randperm(B, device=device)

        # L-swap: shuffle L across the batch, hold V / A / proprio fixed.
        l_swap = model.encode_L(lang_feat[perm], no_drop)
        z_lv_lswap, _, z_la_lswap = _zs(l_swap, v, a, action_kpm)
        p3 = probes.cosine_drift(z_lv, z_lv_lswap)
        lswap_la = probes.cosine_drift(z_la, z_la_lswap)

        # Null-L: replace L by the learned null token.
        drop_all = torch.ones(B, dtype=torch.bool, device=device)
        l_null = model.encode_L(lang_feat, drop_all)
        z_lv_null, _, _ = _zs(l_null, v, a, action_kpm)
        p5_null = probes.cosine_drift(z_lv, z_lv_null)
        p5_ratio = p3 / max(p5_null, 1e-6)

        # P6 (paraphrase distance) only available when the dataloader emits
        # the per-paraphrase tensor. Two paraphrases of the same instruction
        # should land close together if the fuser has semantic grounding.
        p6_para = float("nan")
        lang_raw = batch.get("lang_feat_raw")
        if lang_raw is not None:
            lang_raw = lang_raw.to(device)
            l_p0 = model.encode_L(lang_raw[:, 0], no_drop)
            l_p1 = model.encode_L(lang_raw[:, 1], no_drop)
            z_lv_p0, _, _ = _zs(l_p0, v, a, action_kpm)
            z_lv_p1, _, _ = _zs(l_p1, v, a, action_kpm)
            p6_para = probes.cosine_drift(z_lv_p0, z_lv_p1)

        p7_swap_null = probes.cosine_drift(z_lv_lswap, z_lv_null)

        # V-swap on LV and VA.
        v_swap = model.encode_V(v_raw[perm])
        z_lv_vswap, z_va_vswap, _ = _zs(l, v_swap, a, action_kpm)
        vswap_lv = probes.cosine_drift(z_lv, z_lv_vswap)
        vswap_va = probes.cosine_drift(z_va, z_va_vswap)

        # A-swap on VA and LA: shuffle the encoded A (already carrying
        # proprio); permute the matching pad mask alongside it.
        a_swap = a[perm]
        akpm_swap = action_kpm[perm]
        _, z_va_aswap, z_la_aswap = _zs(l, v, a_swap, akpm_swap)
        aswap_va = probes.cosine_drift(z_va, z_va_aswap)
        aswap_la = probes.cosine_drift(z_la, z_la_aswap)

        # P-swap: shuffle proprio while keeping actions and masks aligned per
        # item -- isolates the proprio-only contribution to z_va / z_la.
        proprio_swap = proprio[perm]
        a_pswap = model.encode_A(act_feats, proprio_swap)
        _, z_va_pswap, z_la_pswap = _zs(l, v, a_pswap, action_kpm)
        pswap_va = probes.cosine_drift(z_va, z_va_pswap)
        pswap_la = probes.cosine_drift(z_la, z_la_pswap)

        cos_lv_va = F.cosine_similarity(z_lv, z_va, dim=-1).mean().item()
        cos_lv_la = F.cosine_similarity(z_lv, z_la, dim=-1).mean().item()
        cos_va_la = F.cosine_similarity(z_va, z_la, dim=-1).mean().item()
    model.train()

    return {
        "p1_max": max(p1), "p2_vals": p2, "p3": p3, "p4": p4,
        "p5_null": p5_null, "p5_ratio": p5_ratio,
        "p6_para": p6_para, "p7_swap_null": p7_swap_null,
        "lswap_la": lswap_la,
        "vswap_lv": vswap_lv, "vswap_va": vswap_va,
        "aswap_va": aswap_va, "aswap_la": aswap_la,
        "pswap_va": pswap_va, "pswap_la": pswap_la,
        "cos_lv_va": cos_lv_va, "cos_lv_la": cos_lv_la, "cos_va_la": cos_va_la,
    }


# ── optimizer + LR schedule ──────────────────────────────────────────────────

def build_optimizer(model: ThreeWayJEPA, cfg: dict) -> torch.optim.Optimizer:
    t = cfg["train"]

    def trainable(params):
        return [p for p in params if p.requires_grad]

    encoder_params = (
        trainable(model.act_embed.parameters())
        + trainable(model.proj_L.parameters())
        + trainable(model.proj_V.parameters())
        + trainable(model.proj_V_ln.parameters())
        + trainable(model.type_emb.parameters())
        + trainable(model.fuser.parameters())
        + ([model.null_L] if model.null_L.requires_grad else [])
    )
    head_params = (
        trainable(model.projector.parameters())
        + trainable(model.predictor.parameters())
    )
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": t["lr_encoders"]},
            {"params": head_params,    "lr": t["lr_heads"]},
        ],
        betas=tuple(t["betas"]),
        weight_decay=t["wd"],
    )


def set_lrs(opt: torch.optim.Optimizer, step: int, cfg: dict) -> None:
    t = cfg["train"]
    base = [t["lr_encoders"], t["lr_heads"]]
    for g, b in zip(opt.param_groups, base):
        g["lr"] = lr_with_warmup_cosine(step, b, t["warmup_steps"], t["total_steps"])


# ── checkpoint I/O ───────────────────────────────────────────────────────────

def save_checkpoint(ckpt_dir: str, step: int,
                    core: ThreeWayJEPA, ema: nn.ModuleDict,
                    opt: torch.optim.Optimizer, cfg: dict,
                    tag: str = "step") -> str:
    """Atomic save to ``<ckpt_dir>/<tag>_<step:07d>.pt``."""
    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"{tag}_{step:07d}.pt")
    tmp_path = final_path + ".tmp"
    torch.save(
        {
            "step": int(step),
            "model": core.state_dict(),
            "ema":   ema.state_dict(),
            "opt":   opt.state_dict(),
            "cfg":   cfg,
        },
        tmp_path,
    )
    os.replace(tmp_path, final_path)
    logger.info(f"[ckpt] saved {final_path}")
    return final_path


def _latest_ckpt(ckpt_dir: str) -> Optional[str]:
    import glob
    if not os.path.isdir(ckpt_dir):
        return None
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")))
    return cands[-1] if cands else None


def load_checkpoint_if_present(ckpt_dir: str,
                               core: ThreeWayJEPA, ema: nn.ModuleDict,
                               opt: torch.optim.Optimizer,
                               device: torch.device) -> int:
    """Resume from the latest ``step_*.pt`` if any. Returns the step to start
    training at (0 when nothing is found)."""
    path = _latest_ckpt(ckpt_dir)
    if path is None:
        return 0
    logger.info(f"[ckpt] resuming from {path}")
    blob = torch.load(path, map_location=device)
    core.load_state_dict(blob["model"])
    ema.load_state_dict(blob["ema"])
    try:
        opt.load_state_dict(blob["opt"])
    except Exception as e:
        logger.warning(f"[ckpt] optimizer state mismatch, continuing fresh: {e}")
    return int(blob["step"]) + 1


# ── logging helper ───────────────────────────────────────────────────────────

def log_metrics(step: int, metrics: dict) -> None:
    parts = [f"step={step}"]
    for k, v in metrics.items():
        parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
    logger.info(" ".join(parts))
    if os.environ.get("JEPA3WAY_WANDB") == "1":
        try:
            import wandb
            wandb.log({**metrics, "step": step})
        except Exception:
            pass


# ── main loop ────────────────────────────────────────────────────────────────

def main(cfg: dict, train_loader: DataLoader, probe_loader: DataLoader,
         device: torch.device, max_steps: Optional[int] = None,
         out_dir: Optional[str] = None,
         save_every: int = 20000,
         save_milestones: Optional[list[int]] = None) -> None:
    logger.info("[boot] building model")
    model = ThreeWayJEPA(cfg).to(device)
    logger.info("[boot] model built; building EMA")
    ema = build_ema_bundle(model).to(device)
    logger.info("[boot] building action registry + optimizer")
    registry = default_registry()
    opt = build_optimizer(model, cfg)
    logger.info("[boot] optimizer built")

    is_ddp = torch.distributed.is_initialized()
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
        core = model.module
    else:
        core = model

    total_steps = max_steps if max_steps is not None else cfg["train"]["total_steps"]
    probe_every = cfg["train"]["probe_every"]

    ckpt_dir: Optional[str] = None
    start_step = 0
    if out_dir is not None:
        ckpt_dir = os.path.join(out_dir, "ckpts")
        start_step = load_checkpoint_if_present(ckpt_dir, core, ema, opt, device)
        if start_step > 0:
            logger.info(f"[ckpt] resumed at step {start_step}")
    milestones = set(save_milestones or [])

    use_bf16 = bool(cfg["train"].get("bf16", False)) and device.type == "cuda"

    def autocast_ctx():
        if use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    # SIGTERM/SIGINT save-and-exit so scheduler kills / Ctrl-C preserve progress.
    interrupted = {"flag": False}

    def handle_sig(signum, frame):
        logger.warning(f"[signal] received {signum}; will save and exit after this step")
        interrupted["flag"] = True

    if ckpt_dir is not None:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handle_sig)
            except Exception:
                pass  # not settable on some platforms (e.g. inside threads)

    step = start_step
    logger.info("[boot] iter(train_loader) -- spawning workers")
    data_iter = iter(train_loader)
    logger.info("[boot] entering training loop")
    first_batch_seen = False

    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        if not first_batch_seen:
            logger.info("[boot] first batch fetched; running first train_step")
            first_batch_seen = True

        set_lrs(opt, step, cfg)
        with autocast_ctx():
            out = train_step(core, ema, batch, registry, cfg, device)
            loss = out["total"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        opt.step()

        tau = cosine_tau(step, cfg["train"]["total_steps"],
                         cfg["train"]["tau_start"], cfg["train"]["tau_end"])
        ema_update_all(ema, core, tau)

        log_metrics(step, {
            "loss": float(loss.detach().float().item()),
            "L_align": out["L_align"], "L_vic": out["L_vic"],
            "tau": tau,
        })

        if probe_every and step > 0 and step % probe_every == 0:
            probe_batch = next(iter(probe_loader))
            info = run_probes(core, probe_batch, registry, device)
            metrics = {
                "probe_p1_max":       info["p1_max"],
                "probe_p2_max":       max(info["p2_vals"]) if info["p2_vals"] else 0.0,
                "probe_p3":           info["p3"],
                "probe_p4":           info["p4"],
                "probe_p5_null":      info["p5_null"],
                "probe_p5_ratio":     info["p5_ratio"],
                "probe_p6_para":      info["p6_para"],
                "probe_p7_swap_null": info["p7_swap_null"],
            }
            for k in ("vswap_lv", "vswap_va", "aswap_la", "aswap_va",
                      "lswap_la", "pswap_va", "pswap_la",
                      "cos_lv_va", "cos_lv_la", "cos_va_la"):
                metrics[f"probe_{k}"] = info[k]
            log_metrics(step, metrics)

        if ckpt_dir is not None and step > start_step:
            do_save = (save_every > 0 and step % save_every == 0) or (step in milestones)
            if do_save:
                save_checkpoint(ckpt_dir, step, core, ema, opt, cfg)

        if interrupted["flag"]:
            if ckpt_dir is not None:
                save_checkpoint(ckpt_dir, step, core, ema, opt, cfg, tag="interrupt")
            logger.warning(f"[signal] exiting at step {step}")
            return

        step += 1

    if ckpt_dir is not None:
        save_checkpoint(ckpt_dir, step, core, ema, opt, cfg, tag="final")
