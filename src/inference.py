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


def load_image_uint8(path, cache_size: int) -> "torch.Tensor":
    """Read an image file as a CHW uint8 tensor resized to ``cache_size``."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB").resize((cache_size, cache_size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.uint8)
    return torch.from_numpy(np.transpose(a, (2, 0, 1)))


def display_crop(u8_chw: "torch.Tensor", img_size: int) -> np.ndarray:
    """Centre-crop a CHW uint8 tensor to ``img_size`` -> HWC uint8 (for display)."""
    _, h, w = u8_chw.shape
    top, left = (h - img_size) // 2, (w - img_size) // 2
    return u8_chw[:, top:top + img_size, left:left + img_size].permute(1, 2, 0).cpu().numpy()


def true_label_from_path(path) -> Optional[str]:
    """Infer the ground-truth species from a dataset path (None if unknown,
    e.g. for an image the evaluator uploads)."""
    from src.data import infer_species
    return infer_species(str(path))


def list_dataset_images(data_dir, species: Optional[Sequence[str]] = None) -> Dict[str, List]:
    """Map species -> sorted image paths, so an evaluator can browse and pick."""
    from collections import defaultdict
    from src.data import find_image_files

    out: Dict[str, List] = defaultdict(list)
    for p in find_image_files(data_dir):
        sp = true_label_from_path(p)
        if sp and (species is None or sp in species):
            out[sp].append(p)
    return {k: sorted(v) for k, v in sorted(out.items())}


class ModelComparer:
    """Runs **every** trained model on the same images for side-by-side review.

    Built for the live demo: an evaluator hand-picks images and sees each
    model's prediction, confidence, top-k and Grad-CAM evidence together, with
    correctness marked automatically when the true species is known from the path.
    """

    def __init__(self, cfg: Config, device, model_keys: Optional[Sequence[str]] = None,
                 smoke: bool = False):
        self.cfg = cfg
        self.device = device
        self.amp = detect_amp(device)
        self.img_size = cfg.data.img_size
        self.cache_size = cfg.data.cache_size
        self.aug = GPUAugment(cfg.augment, self.img_size, device, training=False)
        self.models: Dict[str, "torch.nn.Module"] = {}
        self.class_names: Optional[List[str]] = None

        for k in (model_keys or cfg.model_list):
            ck = _best_ckpt(cfg, k, smoke)
            if not ck.exists():
                LOG.warning("Skipping '%s' (no checkpoint at %s).", k, ck)
                continue
            self.models[k] = load_trained_model(cfg, k, ck, device)
            if self.class_names is None:
                blob = torch.load(ck, map_location="cpu", weights_only=False)
                self.class_names = blob.get("class_names")

        if not self.models:
            raise FileNotFoundError(
                f"No trained checkpoints found in '{cfg.paths.checkpoint_dir}'. "
                f"Train the models first (notebooks 02-04)."
            )
        if not self.class_names:
            from src.data import SPECIES
            self.class_names = sorted(SPECIES)
        LOG.info("ModelComparer ready with %d model(s): %s",
                 len(self.models), list(self.models))

    def compare(self, inputs, top_k: int = 3, with_gradcam: bool = True,
                batch_size: int = 8, max_images: Optional[int] = None) -> List[Dict]:
        """Return one entry per image holding every model's verdict + Grad-CAM."""
        from src.gradcam import compute_gradcam, overlay_cam

        seq = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        paths = gather_image_paths([str(p) for p in seq])
        if not paths:
            raise FileNotFoundError("No valid image files found in the input.")
        if max_images is not None:
            paths = paths[:max_images]

        results: List[Dict] = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            u8 = torch.stack([load_image_uint8(p, self.cache_size)
                              for p in chunk]).to(self.device)
            x = self.aug(u8)

            per_model = {}
            for key, model in self.models.items():
                with torch.no_grad():
                    with torch.amp.autocast(device_type=self.amp.device.type,
                                            dtype=self.amp.amp_dtype,
                                            enabled=self.amp.use_amp):
                        logits = model(x)
                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                cams = compute_gradcam(model, x.clone())[0] if with_gradcam else None
                per_model[key] = (probs, cams)

            for i, path in enumerate(chunk):
                disp = display_crop(u8[i], self.img_size)
                truth = true_label_from_path(path)
                entry: Dict = {"path": str(path), "image": disp, "true": truth,
                               "models": {}}
                for key, (probs, cams) in per_model.items():
                    order = np.argsort(probs[i])[::-1]
                    pred = self.class_names[int(order[0])]
                    info = {
                        "pred": pred,
                        "confidence": float(probs[i, order[0]]),
                        "topk": [(self.class_names[int(j)], float(probs[i, j]))
                                 for j in order[:top_k]],
                        "correct": None if truth is None else (pred == truth),
                    }
                    if cams is not None:
                        info["gradcam"] = overlay_cam(disp, cams[i])
                    entry["models"][key] = info
                entry["agreement"] = len({m["pred"] for m in entry["models"].values()}) == 1
                results.append(entry)

            del u8, x
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return results


def comparison_summary(items: List[Dict]) -> Dict:
    """Per-model correct/total over the reviewed images, plus agreement rate."""
    if not items:
        return {}
    keys = list(items[0]["models"].keys())
    scored = [it for it in items if it.get("true")]
    out = {}
    for k in keys:
        n_ok = sum(1 for it in scored if it["models"][k]["correct"])
        out[k] = {"correct": n_ok, "total": len(scored),
                  "accuracy": (n_ok / len(scored)) if scored else float("nan")}
    out["_agreement"] = sum(1 for it in items if it["agreement"]) / len(items)
    out["_n_images"] = len(items)
    return out


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
                with_gradcam: bool = False, batch_size: int = 16,
                max_images: Optional[int] = None) -> List[Dict]:
        """Predict for image files and/or folders.

        Images are processed in chunks of ``batch_size`` so pointing this at a
        large folder can never exhaust GPU memory (Grad-CAM needs backprop
        memory, so keep the chunk small). ``max_images`` caps how many files are
        read at all -- handy for a fast live demo.
        """
        paths = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        paths = gather_image_paths([str(p) for p in paths])
        if not paths:
            raise FileNotFoundError("No valid image files found in the input.")
        if max_images is not None:
            paths = paths[:max_images]

        results: List[Dict] = []
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            u8 = torch.stack([self._load_uint8(p) for p in chunk]).to(self.device)
            x = self.aug(u8)
            probs = self._probs(x)

            cams = None
            if with_gradcam:
                from src.gradcam import compute_gradcam
                cams, _ = compute_gradcam(self.models[self.cam_model_key], x.clone())

            for i, path in enumerate(chunk):
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
                if cams is not None:
                    from src.gradcam import overlay_cam
                    item["gradcam"] = overlay_cam(disp, cams[i])
                results.append(item)

            del u8, x
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
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
