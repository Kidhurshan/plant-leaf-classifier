#!/usr/bin/env python
"""Evaluate all three models + the ensemble on the held-out TEST set (once) and
export the centrepiece comparison table plus every report figure.

Produces:
  results/metrics/summary_table.md   (and .csv)
  results/metrics/{model}_test_report.txt, {model}_confusion.csv
  results/figures/*.png  (confusion matrices, per-class F1, TTA gain, combined
                          curves, t-SNE)

Usage
-----
    python scripts/evaluate_all.py
    python scripts/evaluate_all.py --smoke        # uses *_smoke checkpoints
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import viz                               # noqa: E402
from src.augment import GPUAugment                # noqa: E402
from src.config import load_config                # noqa: E402
from src.data import prepare_datasets             # noqa: E402
from src.ensemble import build_ensemble           # noqa: E402
from src.evaluate import (                         # noqa: E402
    collect_predictions, compute_metrics, save_classification_report,
    save_confusion_csv,
)
from src.models import load_trained_model      # noqa: E402
from src.report import write_summary_table       # noqa: E402
from src.tta import compare_tta                    # noqa: E402
from src.utils import (                            # noqa: E402
    LOG, count_parameters, detect_amp, get_device, gpu_report, set_seed,
)


def _ckpt(cfg, key, smoke):
    suffix = "_smoke" if smoke else ""
    return Path(cfg.paths.checkpoint_dir) / f"{key}{suffix}_best.pt"


def _train_time(cfg, key, smoke):
    suffix = "_smoke" if smoke else ""
    p = Path(cfg.paths.metrics_dir) / f"{key}{suffix}_history.csv"
    if not p.exists():
        return float("nan")
    with open(p) as fh:
        return sum(float(r["time_s"]) for r in csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate all models + ensemble.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()
    set_seed(cfg.seed)
    device = get_device()
    gpu_report()
    amp = detect_amp(device)
    fig_dir = Path(cfg.paths.figures_dir)
    met_dir = Path(cfg.paths.metrics_dir)

    datasets, class_names = prepare_datasets(cfg, device, smoke=args.smoke)
    bs = cfg.smoke.batch_size if args.smoke else cfg.train.batch_size
    aug_eval = GPUAugment(cfg.augment, cfg.data.img_size, device, training=False)

    per_model = {}          # key -> dict of results
    histories = {}
    test_probs = {}         # key -> base test probs (for ensemble)
    val_f1 = {}
    test_targets = None

    for key in cfg.model_list:
        print("\n" + "#" * 60)
        print(f"# EVALUATE: {key}")
        print("#" * 60)
        model = load_trained_model(cfg, key, _ckpt(cfg, key, args.smoke), device)

        # Validation macro-F1 -> ensemble weight.
        val_eval = collect_predictions(model, datasets["val"], aug_eval, amp, bs)
        val_m = compute_metrics(val_eval["targets"], val_eval["preds"], class_names)
        val_f1[key] = val_m["macro_f1"]

        # TEST set (used once): with and without TTA.
        tta = compare_tta(model, datasets["test"], aug_eval, amp, bs, class_names)
        test_targets = tta["targets"]
        test_probs[key] = tta["without"]["probs"]

        test_m = compute_metrics(test_targets,
                                 tta["without"]["probs"].argmax(1), class_names)
        save_classification_report(test_m["report"],
                                   met_dir / f"{key}_test_report.txt")
        save_confusion_csv(test_m["confusion_matrix"], class_names,
                           met_dir / f"{key}_confusion.csv")
        viz.plot_confusion_matrix(
            test_m["confusion_matrix"], class_names,
            out_path=fig_dir / f"{key}_confusion.png",
            title=f"Confusion matrix — {viz.display_name(key)}")

        counts = count_parameters(model)
        histories[key] = _load_history(cfg, key, args.smoke)
        per_model[key] = {
            "accuracy": test_m["accuracy"],
            "macro_precision": test_m["macro_precision"],
            "macro_recall": test_m["macro_recall"],
            "macro_f1": test_m["macro_f1"],
            "acc_tta": tta["with"]["accuracy"],
            "f1_tta": tta["with"]["macro_f1"],
            "params": counts["total"],
            "train_time_s": _train_time(cfg, key, args.smoke),
            "per_class_f1": test_m["per_class_f1"],
        }
        del model
        if device.type == "cuda":
            import torch
            torch.cuda.empty_cache()

    # ---- ensemble -------------------------------------------------------- #
    ens = build_ensemble(test_probs, val_f1, test_targets, class_names)
    save_confusion_csv(ens["confusion_matrix"], class_names,
                       met_dir / "ensemble_confusion.csv")
    viz.plot_confusion_matrix(ens["confusion_matrix"], class_names,
                              out_path=fig_dir / "ensemble_confusion.png",
                              title="Confusion matrix — Ensemble")
    per_model["ensemble"] = {
        "accuracy": ens["accuracy"], "macro_precision": ens["macro_precision"],
        "macro_recall": ens["macro_recall"], "macro_f1": ens["macro_f1"],
        "acc_tta": float("nan"), "f1_tta": float("nan"),
        "params": sum(per_model[k]["params"] for k in cfg.model_list),
        "train_time_s": sum(per_model[k]["train_time_s"] for k in cfg.model_list),
        "per_class_f1": ens["per_class_f1"],
    }
    LOG.info("Ensemble weights (∝ val macro-F1): %s", ens["weights"])

    # ---- figures --------------------------------------------------------- #
    viz.plot_per_class_f1(
        {k: per_model[k]["per_class_f1"] for k in cfg.model_list},
        class_names, out_path=fig_dir / "per_class_f1.png")
    viz.plot_tta_gain(
        {k: {"without": per_model[k]["macro_f1"], "with": per_model[k]["f1_tta"]}
         for k in cfg.model_list},
        out_path=fig_dir / "tta_gain.png")
    if all(len(histories[k]) for k in histories):
        viz.plot_combined_curves(histories, out_path=fig_dir / "combined_curves.png")

    # t-SNE on the proposed model's penultimate features.
    proposed = cfg.model_list[-1]
    model = load_trained_model(cfg, proposed, _ckpt(cfg, proposed, args.smoke), device)
    feat = collect_predictions(model, datasets["test"], aug_eval, amp, bs,
                               return_features=True)
    viz.plot_tsne(feat["features"], feat["targets"], class_names,
                  out_path=fig_dir / "tsne_features.png",
                  title=f"t-SNE — {viz.display_name(proposed)} features",
                  seed=cfg.seed)

    # ---- summary table --------------------------------------------------- #
    order = cfg.model_list + ["ensemble"]
    write_summary_table(per_model, order, met_dir, fig_dir)
    print("\nDone. Summary -> results/metrics/summary_table.md / .csv")


def _load_history(cfg, key, smoke):
    suffix = "_smoke" if smoke else ""
    p = Path(cfg.paths.metrics_dir) / f"{key}{suffix}_history.csv"
    if not p.exists():
        return []
    with open(p) as fh:
        rows = []
        for r in csv.DictReader(fh):
            rows.append({
                "train_loss": float(r["train_loss"]),
                "val_loss": float(r["val_loss"]),
                "val_acc": float(r["val_acc"]),
                "val_macro_f1": float(r["val_macro_f1"]),
            })
        return rows


if __name__ == "__main__":
    main()
