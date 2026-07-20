"""Self-contained inference for the live demo.

Loads one or more trained checkpoints (never any training state), preprocesses
arbitrary image files exactly like the eval pipeline, and returns predictions
with confidence, top-k probabilities and an optional Grad-CAM overlay.

Both ``scripts/predict.py`` and ``notebooks/06_live_demo.ipynb`` are thin drivers
over :class:`Predictor`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from src.augment import GPUAugment
from src.config import Config
from src.data import IMAGE_EXTS
from src.ensemble import normalise_weights
from src.models import load_trained_model
from src.utils import LOG, detect_amp


def _best_ckpt(cfg: Config, key: str, smoke: bool) -> Path:
    suffix = "_smoke" if smoke else ""
    return Path(cfg.paths.checkpoint_dir) / f"{key}{suffix}_best.pt"


def _read_val_f1(cfg: Config, key: str, smoke: bool) -> float:
    """Read a model's validation macro-F1 from its run-meta (for ensemble
    weights / 'best' selection); falls back to the checkpoint, then 1.0."""
    suffix = "_smoke" if smoke else ""
    meta = Path(cfg.paths.metrics_dir) / f"{key}{suffix}_run_meta.json"
    if meta.exists():
        try:
            return float(json.loads(meta.read_text())["best_val_macro_f1"])
        except Exception:  # noqa: BLE001
            pass
    ck = _best_ckpt(cfg, key, smoke)
    if ck.exists():
        try:
            return float(torch.load(ck, map_location="cpu",
                                    weights_only=False).get("val_macro_f1", 1.0))
        except Exception:  # noqa: BLE001
            pass
    return 1.0


def gather_image_paths(inputs: Sequence[str]) -> List[Path]:
    """Expand a mix of files and directories into a sorted list of image paths."""
    paths: List[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(q for q in p.rglob("*")
                                if q.suffix.lower() in IMAGE_EXTS))
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            paths.append(p)
        else:
            LOG.warning("Skipping '%s' (not an image file or directory).", item)
    return paths


class Predictor:
    """Holds one or more models and runs confidence-weighted inference."""

    def __init__(self, cfg: Config, device, models: Dict[str, "torch.nn.Module"],
                 weights: Dict[str, float], class_names: List[str],
                 cam_model_key: Optional[str] = None):
        self.cfg = cfg
        self.device = device
        self.models = models
        self.weights = normalise_weights(weights)
        self.class_names = class_names
        self.img_size = cfg.data.img_size
        self.cache_size = cfg.data.cache_size
        self.amp = detect_amp(device)
        self.aug = GPUAugment(cfg.augment, self.img_size, device, training=False)
        self.cam_model_key = cam_model_key or next(iter(models))

    # -- preprocessing ------------------------------------------------------ #
    def _load_uint8(self, path: Path) -> torch.Tensor:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB").resize(
                (self.cache_size, self.cache_size), Image.BILINEAR)
            a = np.asarray(im, dtype=np.uint8)
        return torch.from_numpy(np.transpose(a, (2, 0, 1)))  # CHW uint8

    def _display_image(self, u8_chw: torch.Tensor) -> np.ndarray:
        s = self.img_size
        _, h, w = u8_chw.shape
        top, left = (h - s) // 2, (w - s) // 2
        crop = u8_chw[:, top:top + s, left:left + s]
        return crop.permute(1, 2, 0).cpu().numpy()

    # -- prediction --------------------------------------------------------- #
    @torch.no_grad()
    def _probs(self, x: torch.Tensor) -> np.ndarray:
        total = None
        for key, model in self.models.items():
            with torch.amp.autocast(device_type=self.amp.device.type,
                                    dtype=self.amp.amp_dtype, enabled=self.amp.use_amp):
                logits = model(x)
            p = torch.softmax(logits.float(), dim=1) * self.weights[key]
            total = p if total is None else total + p
        return total.cpu().numpy()

    def predict(self, inputs: Sequence[str], top_k: int = 3,
                with_gradcam: bool = False) -> List[Dict]:
        paths = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        paths = gather_image_paths([str(p) for p in paths])
        if not paths:
            raise FileNotFoundError("No valid image files found in the input.")

        u8 = torch.stack([self._load_uint8(p) for p in paths]).to(self.device)
        x = self.aug(u8)
        probs = self._probs(x)

        cams = None
        if with_gradcam:
            from src.gradcam import compute_gradcam
            cam_model = self.models[self.cam_model_key]
            cams, _ = compute_gradcam(cam_model, x.clone())

        results: List[Dict] = []
        for i, path in enumerate(paths):
            order = np.argsort(probs[i])[::-1]
            top = [(self.class_names[int(j)], float(probs[i, j]))
                   for j in order[:top_k]]
            disp = self._display_image(u8[i])
            item = {
                "path": str(path),
                "image": disp,
                "pred": self.class_names[int(order[0])],
                "confidence": float(probs[i, order[0]]),
                "topk": top,
                "probs": probs[i],
            }
            if with_gradcam and cams is not None:
                from src.gradcam import overlay_cam
                item["gradcam"] = overlay_cam(disp, cams[i])
            results.append(item)
        return results


def build_predictor(cfg: Config, device, which: str = "best",
                    smoke: bool = False) -> Predictor:
    """Construct a :class:`Predictor`.

    ``which`` is ``'best'`` (single highest-val-F1 model), ``'ensemble'`` (all
    models with a checkpoint), or a specific model key. Fails gracefully with a
    clear message if the required checkpoint(s) are missing.
    """
    available = [k for k in cfg.model_list if _best_ckpt(cfg, k, smoke).exists()]
    if not available:
        raise FileNotFoundError(
            "No trained checkpoints found in "
            f"'{cfg.paths.checkpoint_dir}'. Train models first (notebooks 02-04 "
            "or scripts/train.py), then retry the demo."
        )

    if which == "ensemble":
        keys = available
    elif which == "best":
        keys = [max(available, key=lambda k: _read_val_f1(cfg, k, smoke))]
    else:
        if _best_ckpt(cfg, which, smoke).exists():
            keys = [which]
        else:
            raise FileNotFoundError(
                f"No checkpoint for '{which}'. Available: {available}.")

    models, weights, class_names = {}, {}, None
    for k in keys:
        m = load_trained_model(cfg, k, _best_ckpt(cfg, k, smoke), device)
        models[k] = m
        weights[k] = _read_val_f1(cfg, k, smoke)
        if class_names is None:
            ck = torch.load(_best_ckpt(cfg, k, smoke), map_location="cpu",
                            weights_only=False)
            class_names = ck.get("class_names")

    if not class_names:
        from src.data import SPECIES
        class_names = sorted(SPECIES)
        LOG.warning("Checkpoint had no class_names; using default %s", class_names)

    # Prefer the CBAM/proposed model for Grad-CAM if it is loaded.
    cam_key = cfg.model_list[-1] if cfg.model_list[-1] in models else keys[0]
    LOG.info("Predictor ready: models=%s, weights=%s", list(models),
             normalise_weights(weights))
    return Predictor(cfg, device, models, weights, list(class_names),
                     cam_model_key=cam_key)
