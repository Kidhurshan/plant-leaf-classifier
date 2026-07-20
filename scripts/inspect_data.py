#!/usr/bin/env python
"""Download the EgyPLI dataset (if needed) and print a full inspection report:
real directory tree, per-folder counts, image dimensions/formats, corrupt files,
and the final class-to-count mapping. Adapts to the actual folder layout.

Usage
-----
    python scripts/inspect_data.py                 # default config
    python scripts/inspect_data.py --no-download    # inspect what is already there
"""
from __future__ import annotations

import argparse
import os
import sys

# Make ``src`` importable when run as a bare script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config           # noqa: E402
from src.data import download_dataset, inspect_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect the EgyPLI dataset.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-download", action="store_true",
                    help="skip the Kaggle download step")
    ap.add_argument("--sample-props", type=int, default=None,
                    help="limit dimension/format scan to N images (speed)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if not args.no_download:
        download_dataset(cfg.data.kaggle_slug, cfg.paths.data_dir)

    inspect_dataset(
        cfg.paths.data_dir,
        expected_classes=cfg.data.num_classes,
        expected_total=cfg.data.expected_total,
        sample_props=args.sample_props,
    )


if __name__ == "__main__":
    main()
