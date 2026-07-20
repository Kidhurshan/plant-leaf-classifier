"""Grad-CAM overlays.

Grad-CAM is computed on the final feature map that :class:`~src.models.LeafClassifier`
already exposes via ``forward_feature_map`` (for the CBAM model this map is
*post*-CBAM). Because the classifier is cleanly split into
``forward_feature_map`` -> ``head``, the same code works for all three backbones.

We make the input a leaf requiring grad so the target feature map always has a
gradient regardless of which backbone params are frozen.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.augment import GPUAugment
from src.utils import LOG


def compute_gradcam(model, x: torch.Tensor,
                    target: Optional[torch.Tensor] = None):
    """Compute Grad-CAM heatmaps for a normalised batch ``x`` [B,3,H,W].

    Returns ``(cam, preds)`` where ``cam`` is ``[B, H, W]`` in ``[0, 1]``
    (upsampled to the input resolution) and ``preds`` is ``[B]``.
    """
    model.eval()
    x = x.clone().requires_grad_(True)
    with torch.enable_grad():
        fmap = model.forward_feature_map(x)      # [B, C, h, w]
        fmap.retain_grad()
        logits = model.head(fmap)
        preds = logits.argmax(1)
        tgt = preds if target is None else target
        score = logits.gather(1, tgt.view(-1, 1)).sum()
        model.zero_grad(set_to_none=True)
        score.backward()

    grads = fmap.grad                            # [B, C, h, w]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * fmap).sum(dim=1)            # [B, h, w]
    cam = F.relu(cam)
    cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:],
                        mode="bilinear", align_corners=False).squeeze(1)
    # Per-sample min-max normalise.
    b = cam.shape[0]
    flat = cam.view(b, -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    cam = (flat - lo) / (hi - lo + 1e-6)
    cam = cam.view(b, *x.shape[-2:])
    return cam.detach().cpu().numpy(), preds.detach().cpu().numpy()


def overlay_cam(image_hwc_uint8: np.ndarray, cam: np.ndarray,
                alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap over an HWC uint8 image using the 'jet' colormap."""
    import matplotlib.cm as cm

    heat = cm.get_cmap("jet")(np.clip(cam, 0, 1))[..., :3]  # [H, W, 3] float
    heat = (heat * 255).astype(np.float32)
    base = image_hwc_uint8.astype(np.float32)
    if base.shape[:2] != heat.shape[:2]:  # safety: resize heat to image
        from PIL import Image
        heat_img = Image.fromarray(heat.astype(np.uint8)).resize(
            (base.shape[1], base.shape[0]))
        heat = np.asarray(heat_img, dtype=np.float32)
    out = (1 - alpha) * base + alpha * heat
    return np.clip(out, 0, 255).astype(np.uint8)


def _center_crop_uint8(img_chw_u8: torch.Tensor, size: int) -> np.ndarray:
    """Centre-crop a CHW uint8 tensor to ``size`` and return HWC uint8."""
    _, h, w = img_chw_u8.shape
    if h < size or w < size:
        img = F.interpolate(img_chw_u8.float().unsqueeze(0),
                            size=(max(h, size), max(w, size)),
                            mode="bilinear", align_corners=False).squeeze(0)
        _, h, w = img.shape
    else:
        img = img_chw_u8.float()
    top, left = (h - size) // 2, (w - size) // 2
    img = img[:, top:top + size, left:left + size]
    return img.clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()


def generate_gradcam_examples(
    model, dataset, class_names: Sequence[str], augment: GPUAugment,
    img_size: int, out_dir: str | Path, per_class: int = 1,
    save: bool = True,
) -> List[Dict]:
    """For each class, find a correct and an incorrect test prediction, compute a
    Grad-CAM overlay, optionally save the PNG, and return items for a viz grid.
    """
    from PIL import Image

    out_dir = Path(out_dir)
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
    device = augment.device
    images = dataset.images
    labels = dataset.labels
    idx = dataset.indices.cpu().numpy()

    # One forward pass to get predictions for the whole test split.
    from src.evaluate import collect_predictions
    from src.utils import detect_amp
    amp = detect_amp(device)
    pred = collect_predictions(model, dataset, augment, amp, batch_size=64)
    preds, targets = pred["preds"], pred["targets"]

    items: List[Dict] = []
    for c in range(len(class_names)):
        pos = np.where(targets == c)[0]
        correct = pos[preds[pos] == c]
        wrong = pos[preds[pos] != c]
        chosen = []
        if len(correct):
            chosen.append(("correct", correct[:per_class]))
        if len(wrong):
            chosen.append(("kind_incorrect", wrong[:per_class]))
        for tag, sel in chosen:
            for local_i in sel:
                global_i = int(idx[local_i])
                img_u8 = images[global_i]                      # CHW uint8 on device
                x = augment(img_u8.unsqueeze(0))               # [1,3,S,S] normalised
                cam, p = compute_gradcam(model, x)
                base = _center_crop_uint8(img_u8, img_size)
                overlay = overlay_cam(base, cam[0])
                true_name = class_names[c]
                pred_name = class_names[int(preds[local_i])]
                item = {"overlay": overlay, "true": true_name, "pred": pred_name,
                        "correct": int(preds[local_i]) == c, "class": true_name}
                items.append(item)
                if save:
                    fname = f"{true_name}_{tag}_pred-{pred_name}.png"
                    Image.fromarray(overlay).save(out_dir / fname)
    if save:
        LOG.info("Saved %d Grad-CAM overlays -> %s", len(items), out_dir)
    return items
