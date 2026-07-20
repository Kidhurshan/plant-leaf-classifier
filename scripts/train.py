#!/usr/bin/env python
"""Train one or all models with the shared engine.

Usage
-----
    python scripts/train.py --model cbam_convnext          # one model, full run
    python scripts/train.py --model all                     # all three
    python scripts/train.py --model all --smoke             # mandatory first run
    python scripts/train.py --model swin_small --no-resume

``--smoke`` uses the ~200-image smoke cache/splits, 1+1 epochs, all models, in
under 5 minutes -- prove the pipeline before spending real compute.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config           # noqa: E402
from src.data import prepare_datasets         # noqa: E402
from src.engine import train_model            # noqa: E402
from src.utils import (                        # noqa: E402
    LOG, Timer, get_device, gpu_report, human_time, set_seed,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train leaf-classification models.")
    ap.add_argument("--model", default="all",
                    help="model key (efficientnetv2s|swin_small|cbam_convnext) or 'all'")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="fast end-to-end pipeline check (~200 images, 1+1 epochs)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any existing checkpoint and train from scratch")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()
    set_seed(cfg.seed)
    device = get_device()
    gpu_report()

    if args.model == "all":
        keys = list(cfg.model_list)
    else:
        if args.model not in cfg.models:
            ap.error(f"Unknown model '{args.model}'. Choose from "
                     f"{list(cfg.models)} or 'all'.")
        keys = [args.model]

    datasets, class_names = prepare_datasets(cfg, device, smoke=args.smoke)
    LOG.info("Datasets ready: train=%d val=%d test=%d | classes=%s",
             len(datasets["train"]), len(datasets["val"]),
             len(datasets["test"]), class_names)

    results = {}
    overall = Timer()
    for key in keys:
        print("\n" + "#" * 70)
        print(f"# TRAINING: {key}  (smoke={args.smoke})")
        print("#" * 70)
        t = Timer()
        res = train_model(cfg, key, datasets, device, smoke=args.smoke,
                          resume=not args.no_resume, class_names=class_names)
        res["train_time_s"] = t.stop()
        results[key] = res
        print(f"\n>> {key}: best val macro-F1 = {res['best_val_macro_f1']:.4f} "
              f"in {human_time(res['train_time_s'])}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for key, res in results.items():
        print(f"  {key:<18} best val macro-F1 = {res['best_val_macro_f1']:.4f}  "
              f"({human_time(res['train_time_s'])})  ckpt={res['best_checkpoint']}")
    print(f"\nTotal wall time: {human_time(overall.stop())}")
    if args.smoke:
        print("\nSMOKE TEST COMPLETE. If this finished cleanly, the full pipeline "
              "is wired correctly — tag v0.1-smoke-passing and start real runs.")


if __name__ == "__main__":
    main()
