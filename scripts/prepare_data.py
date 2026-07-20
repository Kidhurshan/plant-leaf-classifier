#!/usr/bin/env python
"""Build the cached uint8 image tensor and the stratified 70/15/15 split.

The dataset is decoded + resized to ``cache_size`` exactly once and stored as a
single tensor (~705 MB for the full set). The split is written to
``results/metrics/splits.csv`` so all three models see identical data.

Usage
-----
    python scripts/prepare_data.py                 # full cache + splits
    python scripts/prepare_data.py --smoke          # ~200-image smoke cache
    python scripts/prepare_data.py --no-download
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config           # noqa: E402
from src.data import (                        # noqa: E402
    build_cache,
    cache_path,
    discover_dataset,
    download_dataset,
    make_stratified_splits,
    stratified_subset_indices,
    write_splits_csv,
)
from src.utils import LOG                      # noqa: E402


def splits_csv_path(cfg, smoke: bool) -> str:
    name = "splits_smoke.csv" if smoke else "splits.csv"
    return os.path.join(cfg.paths.metrics_dir, name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache the dataset + write splits.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="build a small stratified subset for the smoke test")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the cache already exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()

    if not args.no_download:
        download_dataset(cfg.data.kaggle_slug, cfg.paths.data_dir)

    disc = discover_dataset(cfg.paths.data_dir)
    LOG.info("Discovered %d images across %d classes: %s",
             disc.total, disc.num_classes, disc.class_names)
    assert disc.num_classes == cfg.data.num_classes, (
        f"Expected {cfg.data.num_classes} classes, found {disc.num_classes}."
    )
    if disc.total != cfg.data.expected_total:
        LOG.warning("Total images = %d (expected %d).", disc.total,
                    cfg.data.expected_total)

    out_cache = cache_path(cfg.paths.cache_dir, smoke=args.smoke)
    if out_cache.exists() and not args.force:
        LOG.info("Cache already exists at %s (use --force to rebuild).", out_cache)
    else:
        if args.smoke:
            subset = stratified_subset_indices(
                disc.labels, cfg.smoke.n_images, cfg.seed
            )
            LOG.info("Smoke subset: %d images (stratified).", len(subset))
            build_cache(disc, cfg.data.cache_size, out_cache, subset_idx=subset)
            sub_labels = disc.labels[subset]
            sub_paths = [str(disc.paths[i]) for i in subset]
        else:
            build_cache(disc, cfg.data.cache_size, out_cache)
            sub_labels = disc.labels
            sub_paths = [str(p) for p in disc.paths]

        split = make_stratified_splits(
            sub_labels,
            (cfg.data.split.train, cfg.data.split.val, cfg.data.split.test),
            cfg.seed,
        )
        write_splits_csv(
            sub_paths, sub_labels, disc.class_names, split,
            splits_csv_path(cfg, args.smoke),
        )

    print("\nDone. Cache:", out_cache)
    print("Splits:", splits_csv_path(cfg, args.smoke))


if __name__ == "__main__":
    main()
