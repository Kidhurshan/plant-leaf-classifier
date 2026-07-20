"""Evaluation: prediction collection, metrics, classification report, confusion
matrix (CSV + PNG), and penultimate-feature extraction for t-SNE.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
)

from src.augment import GPUAugment
from src.utils import AmpConfig, LOG


@torch.no_grad()
def collect_predictions(
    model, dataset, augment: GPUAugment, amp: AmpConfig, batch_size: int,
    return_features: bool = False,
):
    """Run the model over a dataset and return softmax probs, preds, targets and
    (optionally) penultimate features (the pooled feature vector before the head
    LayerNorm)."""
    model.eval()
    probs_all: List[np.ndarray] = []
    tgts_all: List[np.ndarray] = []
    feats_all: List[np.ndarray] = []
    for xb_u8, yb in dataset.loader(batch_size, shuffle=False):
        x = augment(xb_u8)
        with torch.amp.autocast(device_type=amp.device.type,
                                dtype=amp.amp_dtype, enabled=amp.use_amp):
            if return_features:
                fmap = model.forward_feature_map(x)
                feats = fmap.mean(dim=(2, 3))         # penultimate features
                logits = model.head(fmap)
            else:
                logits = model(x)
        probs = torch.softmax(logits.float(), dim=1)
        probs_all.append(probs.cpu().numpy())
        tgts_all.append(yb.cpu().numpy())
        if return_features:
            feats_all.append(feats.float().cpu().numpy())
    probs = np.concatenate(probs_all)
    targets = np.concatenate(tgts_all)
    preds = probs.argmax(1)
    out = {"probs": probs, "preds": preds, "targets": targets}
    if return_features:
        out["features"] = np.concatenate(feats_all)
    return out


def compute_metrics(targets, preds, class_names: Sequence[str]) -> Dict:
    """Overall accuracy, per-class + macro precision/recall/F1, classification
    report string, and the confusion matrix."""
    targets = np.asarray(targets)
    preds = np.asarray(preds)
    labels = list(range(len(class_names)))
    p, r, f1, support = precision_recall_fscore_support(
        targets, preds, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        targets, preds, labels=labels, average="macro", zero_division=0
    )
    acc = float((preds == targets).mean())
    cm = confusion_matrix(targets, preds, labels=labels)
    report = classification_report(
        targets, preds, labels=labels, target_names=list(class_names),
        zero_division=0, digits=4,
    )
    return {
        "accuracy": acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm,
        "report": report,
        "class_names": list(class_names),
    }


def save_confusion_csv(cm, class_names: Sequence[str], path: str | Path) -> Path:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(cm)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred", *class_names])
        for i, name in enumerate(class_names):
            w.writerow([name, *cm[i].tolist()])
    return path


def save_classification_report(report: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    return path


def evaluate_model(
    model, dataset, augment: GPUAugment, amp: AmpConfig, batch_size: int,
    class_names: Sequence[str], return_features: bool = False,
) -> Dict:
    """Convenience: collect predictions + compute all metrics in one call."""
    pred = collect_predictions(model, dataset, augment, amp, batch_size,
                               return_features=return_features)
    metrics = compute_metrics(pred["targets"], pred["preds"], class_names)
    metrics.update({k: pred[k] for k in ("probs", "preds", "targets")})
    if return_features:
        metrics["features"] = pred["features"]
    LOG.info("Eval: acc=%.4f macro-F1=%.4f", metrics["accuracy"], metrics["macro_f1"])
    return metrics
