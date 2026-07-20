#!/usr/bin/env python
"""Live-demo prediction CLI.

Loads the best model (or the ensemble) from saved checkpoints only, then predicts
the species + confidence + top-3 for each image path or folder given.

Usage
-----
    python scripts/predict.py --images path/to/img.jpg
    python scripts/predict.py --images folder/ --model ensemble --topk 3
    python scripts/predict.py --images a.jpg b.jpg --gradcam --save-dir results/gradcam
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config           # noqa: E402
from src.inference import build_predictor     # noqa: E402
from src.utils import get_device              # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict leaf species from images.")
    ap.add_argument("--images", nargs="+", required=True,
                    help="image file(s) and/or folder(s)")
    ap.add_argument("--model", default="best",
                    help="'best' | 'ensemble' | a model key")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--gradcam", action="store_true")
    ap.add_argument("--save-dir", default=None,
                    help="if given with --gradcam, save overlay PNGs here")
    ap.add_argument("--smoke", action="store_true",
                    help="use *_smoke checkpoints (for pipeline testing)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    try:
        predictor = build_predictor(cfg, device, which=args.model, smoke=args.smoke)
    except FileNotFoundError as exc:
        print(f"\n[predict] {exc}")
        sys.exit(1)

    results = predictor.predict(args.images, top_k=args.topk,
                                with_gradcam=args.gradcam)

    print("\n" + "=" * 60)
    for r in results:
        print(f"\nImage : {r['path']}")
        print(f"  -> {r['pred']}  ({r['confidence'] * 100:.1f}% confidence)")
        print("  Top-%d:" % args.topk)
        for name, prob in r["topk"]:
            print(f"    {name:<12} {prob * 100:5.1f}%")

    if args.gradcam and args.save_dir:
        from pathlib import Path
        from PIL import Image
        out = Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for r in results:
            if "gradcam" in r:
                name = Path(r["path"]).stem + f"_cam_{r['pred']}.png"
                Image.fromarray(r["gradcam"]).save(out / name)
        print(f"\nGrad-CAM overlays saved to {out}")


if __name__ == "__main__":
    main()
