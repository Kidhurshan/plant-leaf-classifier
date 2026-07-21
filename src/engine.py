"""Shared two-phase training engine.

All three models are trained by *this* code, so the only variable is the
backbone. Features:

* **Phase 1** -- freeze backbone, train head (+CBAM) for a few epochs at a higher LR.
* **Phase 2** -- unfreeze everything, fine-tune at a low LR with layer-wise LR
  decay, cosine schedule and short warmup, mixup/cutmix on (phase 2 only).
* AMP (bf16 where supported, else fp16 + GradScaler), AdamW, grad clipping.
* Early stopping on validation macro-F1 with best-only checkpointing.
* Auto-resume from the last checkpoint; auto-reduce batch size on CUDA OOM.
* Per-epoch history CSV and a run-metadata JSON for full reproducibility.
"""
from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from sklearn.metrics import f1_score
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.augment import GPUAugment, apply_mix
from src.config import Config
from src.data import splits_csv_path
from src.losses import build_loss
from src.models import build_model
from src.utils import (
    AmpConfig, LOG, detect_amp, file_fingerprint, human_time, save_run_meta,
)


# --------------------------------------------------------------------------- #
# Optimiser / scheduler                                                       #
# --------------------------------------------------------------------------- #
def _param_layer_index(name: str) -> Optional[int]:
    m = re.search(r"(?:stages|layers|blocks)\.(\d+)", name)
    return int(m.group(1)) if m else None


def build_param_groups(
    model, base_lr: float, weight_decay: float,
    use_llrd: bool = False, llrd_decay: float = 0.85,
) -> List[dict]:
    """Build AdamW parameter groups.

    * 1-D params (biases, norm weights) get ``weight_decay = 0``.
    * With ``use_llrd``, backbone params get an LR decayed by depth (earlier
      layers -> smaller LR); the head/CBAM keep the full ``base_lr``.
    """
    # Deepest parsed backbone stage index (for the decay exponent).
    max_idx = 0
    for name, p in model.named_parameters():
        if name.startswith("backbone."):
            idx = _param_layer_index(name)
            if idx is not None:
                max_idx = max(max_idx, idx)

    groups: Dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if use_llrd and name.startswith("backbone."):
            idx = _param_layer_index(name)
            if idx is None:
                low = any(k in name for k in
                          ("stem", "patch_embed", "embed", "downsample"))
                idx = 0 if low else max_idx
            lr = base_lr * (llrd_decay ** (max_idx - idx))
        else:
            lr = base_lr
        wd = weight_decay if p.ndim >= 2 else 0.0
        key = (round(lr, 10), wd)
        groups.setdefault(key, {"params": [], "lr": key[0], "weight_decay": key[1]})
        groups[key]["params"].append(p)
    return list(groups.values())


def build_optimizer(model, cfg: Config, phase: int) -> torch.optim.Optimizer:
    if phase == 1:
        groups = build_param_groups(
            model, cfg.train.phase1_lr, cfg.train.weight_decay, use_llrd=False
        )
    else:
        groups = build_param_groups(
            model, cfg.train.phase2_lr, cfg.train.weight_decay,
            use_llrd=cfg.train.use_llrd, llrd_decay=cfg.train.llrd_decay,
        )
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def build_scheduler(optimizer, epochs: int, steps_per_epoch: int, warmup_epochs: int):
    total = max(1, epochs * steps_per_epoch)
    warm = max(0, warmup_epochs) * steps_per_epoch
    if total > 1:
        warm = min(warm, total - 1)
    else:
        warm = 0
    if warm <= 0:
        return CosineAnnealingLR(optimizer, T_max=total)
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=warm),
            CosineAnnealingLR(optimizer, T_max=total - warm),
        ],
        milestones=[warm],
    )


# --------------------------------------------------------------------------- #
# Train / validate steps                                                      #
# --------------------------------------------------------------------------- #
class _OOM(Exception):
    """Internal signal: CUDA OOM -> reduce batch size and retry the phase."""


def _is_oom(err: RuntimeError) -> bool:
    return "out of memory" in str(err).lower()


def train_one_epoch(
    model, dataset, augment: GPUAugment, criterion, optimizer, scaler,
    amp: AmpConfig, batch_size: int, grad_clip: float,
    scheduler=None, aug_cfg=None, use_mix: bool = False,
    epoch_seed: int = 0,
) -> float:
    model.train()
    total, count = 0.0, 0
    for xb_u8, yb in dataset.loader(batch_size, shuffle=True, seed=epoch_seed):
        try:
            x = augment(xb_u8)
            ya, yb2, lam = yb, yb, 1.0
            if use_mix:
                x, ya, yb2, lam = apply_mix(x, yb, aug_cfg)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=amp.device.type,
                                    dtype=amp.amp_dtype, enabled=amp.use_amp):
                logits = model(x)
                if lam < 1.0:
                    loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb2)
                else:
                    loss = criterion(logits, yb)
            if scaler is not None and amp.needs_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        except RuntimeError as err:
            if _is_oom(err):
                optimizer.zero_grad(set_to_none=True)
                if amp.device.type == "cuda":
                    torch.cuda.empty_cache()
                raise _OOM() from err
            raise
        bs = x.size(0)
        total += loss.item() * bs
        count += bs
    return total / max(count, 1)


@torch.no_grad()
def validate(
    model, dataset, augment: GPUAugment, criterion, num_classes: int,
    amp: AmpConfig, batch_size: int,
) -> Dict[str, float]:
    model.eval()
    total, count = 0.0, 0
    all_preds: List[torch.Tensor] = []
    all_tgts: List[torch.Tensor] = []
    for xb_u8, yb in dataset.loader(batch_size, shuffle=False):
        x = augment(xb_u8)
        with torch.amp.autocast(device_type=amp.device.type,
                                dtype=amp.amp_dtype, enabled=amp.use_amp):
            logits = model(x)
            loss = criterion(logits, yb)
        total += loss.item() * x.size(0)
        count += x.size(0)
        all_preds.append(logits.argmax(1).cpu())
        all_tgts.append(yb.cpu())
    preds = torch.cat(all_preds).numpy()
    tgts = torch.cat(all_tgts).numpy()
    acc = float((preds == tgts).mean())
    macro_f1 = float(f1_score(tgts, preds, average="macro", zero_division=0))
    return {"loss": total / max(count, 1), "acc": acc, "macro_f1": macro_f1}


# --------------------------------------------------------------------------- #
# Checkpoint helpers                                                          #
# --------------------------------------------------------------------------- #
def _ckpt_paths(ckpt_dir: Path, model_key: str):
    return ckpt_dir / f"{model_key}_best.pt", ckpt_dir / f"{model_key}_last.pt"


def save_checkpoint(path: Path, model, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), **extra}
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #
def train_model(
    cfg: Config,
    model_key: str,
    datasets: dict,
    device,
    smoke: bool = False,
    resume: bool = True,
    class_names: Optional[List[str]] = None,
) -> dict:
    """Train one model end-to-end (two phases) and return a results dict.

    ``datasets`` must contain ``train`` and ``val`` :class:`GPUTensorDataset`s.
    ``class_names`` (if given) is recorded in the best checkpoint so the demo /
    prediction script is self-contained.
    """
    amp = detect_amp(device)
    num_classes = cfg.data.num_classes
    img_size = cfg.data.img_size
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    best_path, last_path = _ckpt_paths(ckpt_dir, model_key + ("_smoke" if smoke else ""))

    p1_epochs = 1 if smoke else cfg.train.phase1_epochs
    p2_epochs = 1 if smoke else cfg.train.phase2_epochs
    batch_size = cfg.smoke.batch_size if smoke else cfg.train.batch_size

    model = build_model(cfg, model_key, device=device)
    counts = datasets["train"].class_counts(num_classes)
    LOG.info("Train class counts: %s", counts.tolist())
    criterion = build_loss(cfg, counts, device)
    aug_train = GPUAugment(cfg.augment, img_size, device, training=True)
    aug_eval = GPUAugment(cfg.augment, img_size, device, training=False)
    scaler = torch.amp.GradScaler(amp.device.type, enabled=amp.needs_scaler)

    history: List[dict] = []
    state = {"best_f1": -1.0, "epoch_wall": []}

    # Fingerprint of the exact split this run trains on. Stamped into every
    # checkpoint so a checkpoint from a DIFFERENT split is never resumed into.
    split_fp = file_fingerprint(splits_csv_path(cfg.paths.metrics_dir, smoke))

    # ---- resume ----------------------------------------------------------- #
    start_phase, start_epoch = 1, 1
    if resume and last_path.exists():
        ck = load_checkpoint(last_path, map_location=device)
        ck_fp = ck.get("split_fingerprint")
        if ck_fp != split_fp:
            LOG.warning(
                "Ignoring checkpoint %s: it was trained on a DIFFERENT data "
                "split (fingerprint %s != %s). Training '%s' FROM SCRATCH.",
                last_path.name, ck_fp, split_fp, model_key,
            )
            print(f"  >> Split changed since that checkpoint -> training "
                  f"'{model_key}' from scratch (old checkpoint ignored).")
        else:
            model.load_state_dict(ck["model_state"])
            start_phase = ck.get("phase", 1)
            start_epoch = ck.get("epoch", 0) + 1
            state["best_f1"] = ck.get("best_f1", -1.0)
            history = ck.get("history", [])
            LOG.info("Resumed '%s' from %s (phase %d, next epoch %d, "
                     "best_f1=%.4f).", model_key, last_path, start_phase,
                     start_epoch, state["best_f1"])

    def _log_epoch(phase: int, epoch: int, tr_loss: float,
                   val: dict, lr: float, dt: float) -> None:
        row = {"phase": phase, "epoch": epoch, "train_loss": round(tr_loss, 4),
               "val_loss": round(val["loss"], 4), "val_acc": round(val["acc"], 4),
               "val_macro_f1": round(val["macro_f1"], 4), "lr": lr,
               "time_s": round(dt, 1)}
        history.append(row)
        print(f"  [P{phase} E{epoch:02d}] train_loss={tr_loss:.4f} "
              f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} "
              f"val_f1={val['macro_f1']:.4f} lr={lr:.2e} ({human_time(dt)})")

    def _maybe_estimate_runtime(planned_total: int) -> None:
        if len(state["epoch_wall"]) == 1:
            per = state["epoch_wall"][0]
            print(f"  >> Estimated total training time: "
                  f"~{human_time(per * planned_total)} "
                  f"({planned_total} planned epochs @ {human_time(per)}/epoch).")

    def _save_best(phase: int, epoch: int, val: dict) -> None:
        if val["macro_f1"] > state["best_f1"]:
            state["best_f1"] = val["macro_f1"]
            save_checkpoint(best_path, model, {
                "model_key": model_key, "phase": phase, "epoch": epoch,
                "val_macro_f1": val["macro_f1"], "val_acc": val["acc"],
                "class_names": class_names,
                "num_classes": num_classes, "img_size": img_size,
                "use_cbam": cfg.model_def(model_key).cbam,
                "split_fingerprint": split_fp,
                "backbone": model.backbone_name,
            })
            print(f"    * new best macro-F1={val['macro_f1']:.4f} -> saved {best_path.name}")

    planned_total = (p1_epochs + p2_epochs)

    # ---- PHASE 1: frozen backbone ---------------------------------------- #
    if start_phase == 1:
        model.set_backbone_trainable(False)
        opt1 = build_optimizer(model, cfg, phase=1)
        print(f"\n=== {model_key}: PHASE 1 (frozen backbone, {p1_epochs} epochs) ===")
        for epoch in range(start_epoch if start_phase == 1 else 1, p1_epochs + 1):
            t0 = time.perf_counter()
            tr = train_one_epoch(
                model, datasets["train"], aug_train, criterion, opt1, scaler,
                amp, batch_size, cfg.train.grad_clip, scheduler=None,
                aug_cfg=cfg.augment, use_mix=False, epoch_seed=cfg.seed + epoch,
            )
            val = validate(model, datasets["val"], aug_eval, criterion,
                           num_classes, amp, batch_size)
            dt = time.perf_counter() - t0
            state["epoch_wall"].append(dt)
            _log_epoch(1, epoch, tr, val, opt1.param_groups[0]["lr"], dt)
            _save_best(1, epoch, val)
            _maybe_estimate_runtime(planned_total)
            save_checkpoint(last_path, model, {
                "phase": 1, "epoch": epoch, "best_f1": state["best_f1"],
                "history": history, "model_key": model_key,
                "split_fingerprint": split_fp,
            })
        start_phase, start_epoch = 2, 1  # advance

    # ---- PHASE 2: fine-tune all (with OOM auto-reduce) ------------------- #
    print(f"\n=== {model_key}: PHASE 2 (fine-tune all, up to {p2_epochs} epochs) ===")
    bs = batch_size
    while True:
        model.set_backbone_trainable(True)
        opt2 = build_optimizer(model, cfg, phase=2)
        steps_per_epoch = max(1, math.ceil(len(datasets["train"]) / bs))
        sched = build_scheduler(opt2, p2_epochs, steps_per_epoch,
                                0 if smoke else cfg.train.warmup_epochs)
        patience = 0
        use_mix = (cfg.augment.mixup or cfg.augment.cutmix)  # phase-2-only by design
        try:
            for epoch in range(start_epoch, p2_epochs + 1):
                t0 = time.perf_counter()
                tr = train_one_epoch(
                    model, datasets["train"], aug_train, criterion, opt2, scaler,
                    amp, bs, cfg.train.grad_clip, scheduler=sched,
                    aug_cfg=cfg.augment, use_mix=use_mix,
                    epoch_seed=cfg.seed + 100 + epoch,
                )
                val = validate(model, datasets["val"], aug_eval, criterion,
                               num_classes, amp, bs)
                dt = time.perf_counter() - t0
                state["epoch_wall"].append(dt)
                _log_epoch(2, epoch, tr, val, opt2.param_groups[0]["lr"], dt)
                _maybe_estimate_runtime(planned_total)
                improved = val["macro_f1"] > state["best_f1"]
                _save_best(2, epoch, val)
                patience = 0 if improved else patience + 1
                save_checkpoint(last_path, model, {
                    "phase": 2, "epoch": epoch, "best_f1": state["best_f1"],
                    "history": history, "model_key": model_key,
                    "split_fingerprint": split_fp,
                })
                if patience >= cfg.train.patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(no val macro-F1 improvement for {patience} epochs).")
                    break
            break  # phase 2 completed without OOM
        except _OOM:
            new_bs = max(4, bs // 2)
            if new_bs == bs:
                raise RuntimeError("CUDA OOM even at the minimum batch size (4).")
            LOG.warning("CUDA OOM: reducing batch size %d -> %d and restarting "
                        "phase 2.", bs, new_bs)
            bs = new_bs
            start_epoch = 1
            history[:] = [h for h in history if h["phase"] == 1]  # drop aborted P2 rows
            if amp.device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- finalise --------------------------------------------------------- #
    metrics_dir = Path(cfg.paths.metrics_dir)
    suffix = "_smoke" if smoke else ""
    _write_history_csv(history, metrics_dir / f"{model_key}{suffix}_history.csv")
    save_run_meta(
        metrics_dir / f"{model_key}{suffix}_run_meta.json",
        model_key=model_key, seed=cfg.seed, config=cfg.to_dict(),
        extra={"best_val_macro_f1": state["best_f1"],
               "best_checkpoint": str(best_path),
               "batch_size_final": bs, "smoke": smoke,
               "amp_dtype": amp.dtype_name},
    )

    # Load best weights back for immediate evaluation.
    if best_path.exists():
        model.load_state_dict(load_checkpoint(best_path, map_location=device)["model_state"])
        print(f"\nLoaded best checkpoint (val macro-F1={state['best_f1']:.4f}).")

    return {
        "model": model,
        "model_key": model_key,
        "history": history,
        "best_val_macro_f1": state["best_f1"],
        "best_checkpoint": str(best_path),
        "history_csv": str(metrics_dir / f"{model_key}{suffix}_history.csv"),
    }


def read_history_csv(path) -> List[dict]:
    """Load a ``{model}_history.csv`` back into a list of per-epoch dicts."""
    path = Path(path)
    if not path.exists():
        return []
    rows: List[dict] = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "phase": int(r["phase"]), "epoch": int(r["epoch"]),
                "train_loss": float(r["train_loss"]), "val_loss": float(r["val_loss"]),
                "val_acc": float(r["val_acc"]), "val_macro_f1": float(r["val_macro_f1"]),
                "time_s": float(r["time_s"]),
            })
    return rows


def _write_history_csv(history: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fields = ["phase", "epoch", "train_loss", "val_loss", "val_acc",
              "val_macro_f1", "lr", "time_s"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in history:
            w.writerow({k: row.get(k) for k in fields})
    LOG.info("Wrote training history -> %s", path)
