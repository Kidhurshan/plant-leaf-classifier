"""Test-time augmentation: average softmax over {original, hflip, vflip, both}.

Reported with and without so the report can show the gain.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from src.augment import GPUAugment
from src.evaluate import compute_metrics
from src.utils import AmpConfig


_TTA_FLIPS = [
    lambda t: t,                         # original
    lambda t: t.flip(-1),                # horizontal flip
    lambda t: t.flip(-2),                # vertical flip
    lambda t: t.flip(-1).flip(-2),       # both
]


@torch.no_grad()
def tta_predictions(model, dataset, augment: GPUAugment, amp: AmpConfig,
                    batch_size: int) -> Dict[str, np.ndarray]:
    """Return TTA-averaged probs, preds and targets."""
    model.eval()
    probs_all: List[np.ndarray] = []
    tgts_all: List[np.ndarray] = []
    for xb_u8, yb in dataset.loader(batch_size, shuffle=False):
        x = augment(xb_u8)  # deterministic eval preprocessing (centre-crop + norm)
        acc = None
        for flip in _TTA_FLIPS:
            xv = flip(x)
            with torch.amp.autocast(device_type=amp.device.type,
                                    dtype=amp.amp_dtype, enabled=amp.use_amp):
                logits = model(xv)
            p = torch.softmax(logits.float(), dim=1)
            acc = p if acc is None else acc + p
        acc = acc / len(_TTA_FLIPS)
        probs_all.append(acc.cpu().numpy())
        tgts_all.append(yb.cpu().numpy())
    probs = np.concatenate(probs_all)
    targets = np.concatenate(tgts_all)
    return {"probs": probs, "preds": probs.argmax(1), "targets": targets}


def compare_tta(model, dataset, augment: GPUAugment, amp: AmpConfig,
                batch_size: int, class_names) -> Dict:
    """Evaluate the model with and without TTA and return both metric sets plus
    the deltas."""
    from src.evaluate import collect_predictions

    base = collect_predictions(model, dataset, augment, amp, batch_size)
    base_m = compute_metrics(base["targets"], base["preds"], class_names)

    tta = tta_predictions(model, dataset, augment, amp, batch_size)
    tta_m = compute_metrics(tta["targets"], tta["preds"], class_names)

    return {
        "without": {"accuracy": base_m["accuracy"], "macro_f1": base_m["macro_f1"],
                    "probs": base["probs"]},
        "with": {"accuracy": tta_m["accuracy"], "macro_f1": tta_m["macro_f1"],
                 "probs": tta["probs"]},
        "targets": base["targets"],
        "delta_acc": tta_m["accuracy"] - base_m["accuracy"],
        "delta_macro_f1": tta_m["macro_f1"] - base_m["macro_f1"],
    }
