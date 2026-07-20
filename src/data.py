"""Dataset discovery, inspection, caching, stratified splits and a GPU-resident
tensor dataset.

Design notes
------------
* We do **not** assume the EgyPLI folder layout. Species are inferred by matching
  known species keywords against each file's path, which transparently merges
  nested ``healthy``/``diseased`` subfolders into one species label.
* The whole dataset is small, so the loading bottleneck is removed entirely: every
  image is decoded + resized to ``cache_size`` exactly once and stored as a single
  uint8 tensor. At train time that tensor lives on the GPU and is indexed directly;
  augmentation happens on-GPU in batch (see :mod:`src.augment`).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils import LOG, bytes_to_human

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

# The 8 EgyPLI species. Matching is substring-on-path, so variants like
# "Blueberry" still map to "berry" and "Apple___healthy" maps to "apple".
SPECIES: List[str] = [
    "apple", "berry", "fig", "guava",
    "orange", "plum", "persimmon", "tomato",
]

# Tokens that indicate leaf health status, merged away into the species label.
HEALTH_TOKENS = {"healthy", "diseased", "disease", "unhealthy", "sick", "infected"}


# --------------------------------------------------------------------------- #
# Species inference                                                            #
# --------------------------------------------------------------------------- #
def infer_species(rel_path: str, species: Sequence[str] = SPECIES) -> Optional[str]:
    """Infer the species label from a file's path.

    Prefers a whole path-component match (e.g. a folder literally named
    ``apple``); falls back to a substring match so ``blueberry`` -> ``berry``.
    Returns ``None`` if nothing matches (caller warns and drops the file).
    """
    low = rel_path.lower()
    tokens = set(re.split(r"[^a-z]+", low))
    whole = [s for s in species if s in tokens]
    if whole:
        return max(whole, key=len)
    sub = [s for s in species if s in low]
    if sub:
        return max(sub, key=len)
    return None


# --------------------------------------------------------------------------- #
# Discovery                                                                    #
# --------------------------------------------------------------------------- #
def find_image_files(root: str | Path) -> List[Path]:
    """Recursively find every image file under ``root`` (sorted for determinism)."""
    root = Path(root)
    files = [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(files)


@dataclass
class Discovery:
    """Result of scanning the raw dataset directory."""

    root: Path
    paths: List[Path]                 # kept (mapped) image paths
    labels: np.ndarray                # int64 label per kept path
    class_names: List[str]            # sorted species names, index == label id
    class_counts: Dict[str, int]      # species -> count
    unmapped: List[Path] = field(default_factory=list)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def total(self) -> int:
        return len(self.paths)


def discover_dataset(
    root: str | Path, species: Sequence[str] = SPECIES
) -> Discovery:
    """Find all images, infer + merge species labels, and build a label map.

    Species are assigned integer ids in **alphabetical order** for determinism,
    so the mapping is identical across every run and every machine.
    """
    root = Path(root)
    all_files = find_image_files(root)
    if not all_files:
        raise FileNotFoundError(
            f"No image files found under {root.resolve()}. "
            f"Did the Kaggle download succeed?"
        )

    kept_paths: List[Path] = []
    kept_species: List[str] = []
    unmapped: List[Path] = []
    for p in all_files:
        rel = str(p.relative_to(root))
        sp = infer_species(rel, species)
        if sp is None:
            unmapped.append(p)
            continue
        kept_paths.append(p)
        kept_species.append(sp)

    if unmapped:
        LOG.warning(
            "%d/%d files could not be mapped to a species and were dropped "
            "(first few: %s).",
            len(unmapped), len(all_files),
            [str(u.name) for u in unmapped[:5]],
        )

    class_names = sorted(set(kept_species))
    name_to_id = {name: i for i, name in enumerate(class_names)}
    labels = np.array([name_to_id[s] for s in kept_species], dtype=np.int64)
    class_counts = dict(sorted(Counter(kept_species).items()))

    return Discovery(
        root=root,
        paths=kept_paths,
        labels=labels,
        class_names=class_names,
        class_counts=class_counts,
        unmapped=unmapped,
    )


# --------------------------------------------------------------------------- #
# Inspection (used by scripts/inspect_data.py)                                 #
# --------------------------------------------------------------------------- #
def print_directory_tree(root: str | Path, max_entries: int = 40) -> None:
    """Print the real directory tree with a per-folder image count."""
    root = Path(root)
    print(f"\nDirectory tree under: {root.resolve()}")
    dirs_with_counts: List[Tuple[Path, int]] = []
    for d in sorted([root, *[p for p in root.rglob("*") if p.is_dir()]]):
        n = sum(
            1 for c in d.iterdir()
            if c.is_file() and c.suffix.lower() in IMAGE_EXTS
        )
        dirs_with_counts.append((d, n))
    shown = 0
    for d, n in dirs_with_counts:
        depth = len(d.relative_to(root).parts)
        indent = "  " * depth
        tag = f"  [{n} images]" if n else ""
        print(f"{indent}{d.name}/{tag}")
        shown += 1
        if shown >= max_entries:
            print(f"  ... ({len(dirs_with_counts) - shown} more directories)")
            break


def scan_image_properties(
    paths: Sequence[Path], sample: Optional[int] = None
) -> Dict[str, object]:
    """Collect image dimensions, formats and detect corrupt files.

    ``sample`` limits the dimension/format scan for speed; corruption is always
    checked on every file.
    """
    from PIL import Image  # local import: Pillow is a Colab-managed dep

    formats: Counter = Counter()
    sizes: Counter = Counter()
    corrupt: List[str] = []

    rng = np.random.default_rng(0)
    if sample is not None and sample < len(paths):
        probe_idx = set(rng.choice(len(paths), size=sample, replace=False).tolist())
    else:
        probe_idx = set(range(len(paths)))

    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                im.verify()  # cheap integrity check
            if i in probe_idx:
                with Image.open(p) as im2:
                    formats[im2.format or p.suffix.lstrip(".").upper()] += 1
                    sizes[im2.size] += 1
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            corrupt.append(f"{p}  ({exc})")

    return {
        "formats": dict(formats),
        "top_sizes": sizes.most_common(10),
        "corrupt": corrupt,
        "n_scanned_props": len(probe_idx),
    }


def inspect_dataset(
    root: str | Path,
    expected_classes: int = 8,
    expected_total: int = 3588,
    sample_props: Optional[int] = None,
) -> Discovery:
    """Full inspection: tree, counts, dimensions, formats, corrupt files.

    Prints everything, asserts the class count, and *warns loudly* (never
    crashes) if the total image count differs from ``expected_total``.
    """
    root = Path(root)
    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)
    print_directory_tree(root)

    disc = discover_dataset(root)

    print("\nPer-folder image counts (leaf directories):")
    per_folder: Dict[str, int] = defaultdict(int)
    for p in disc.paths:
        per_folder[str(p.parent.relative_to(root))] += 1
    for folder, n in sorted(per_folder.items()):
        print(f"  {folder:<45} {n:>5}")

    print("\nImage properties:")
    props = scan_image_properties(disc.paths, sample=sample_props)
    print(f"  formats        : {props['formats']}")
    print(f"  top dimensions : {props['top_sizes']}")
    if props["corrupt"]:
        print(f"  CORRUPT files  : {len(props['corrupt'])}")
        for c in props["corrupt"][:10]:
            print(f"    - {c}")
    else:
        print("  corrupt files  : none")

    print("\nFinal class-to-count mapping (species label id : name : count):")
    for i, name in enumerate(disc.class_names):
        print(f"  {i}: {name:<12} {disc.class_counts.get(name, 0):>5}")
    print(f"  {'TOTAL':<15} {disc.total:>5}")

    # Assert class count (allowed to crash); warn on total mismatch.
    assert disc.num_classes == expected_classes, (
        f"Expected {expected_classes} classes but found {disc.num_classes}: "
        f"{disc.class_names}. Check species inference / dataset layout."
    )
    if disc.total != expected_total:
        LOG.warning(
            "!!! Total image count is %d, expected %d. Proceeding anyway "
            "(this is a warning, not an error).",
            disc.total, expected_total,
        )
    else:
        print(f"\nOK: {disc.num_classes} classes, {disc.total} images (as expected).")
    return disc


# --------------------------------------------------------------------------- #
# Kaggle download                                                             #
# --------------------------------------------------------------------------- #
def ensure_kaggle_credentials() -> bool:
    """Make Kaggle credentials available to the CLI.

    Order: existing ~/.kaggle/kaggle.json -> env vars -> Colab userdata secrets.
    Writes ~/.kaggle/kaggle.json (chmod 600) if it can source a username+key.
    Returns True if credentials are in place. Never prints the key.
    """
    import json
    import os
    import stat

    kaggle_dir = Path.home() / ".kaggle"
    cred_path = kaggle_dir / "kaggle.json"
    if cred_path.exists():
        try:
            cred_path.chmod(0o600)
        except OSError:
            pass
        return True

    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")

    if not (username and key):
        try:  # Colab secrets
            from google.colab import userdata  # type: ignore
            username = username or userdata.get("KAGGLE_USERNAME")
            key = key or userdata.get("KAGGLE_KEY")
        except Exception:  # noqa: BLE001 - not on Colab / secret not set
            pass

    if username and key:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        cred_path.write_text(json.dumps({"username": username, "key": key}))
        cred_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 600
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        return True

    LOG.error(
        "No Kaggle credentials found. Provide them via KAGGLE_USERNAME/"
        "KAGGLE_KEY env vars, Colab secrets, or upload kaggle.json to "
        "~/.kaggle/kaggle.json (chmod 600)."
    )
    return False


def download_dataset(slug: str, dest: str | Path, force: bool = False) -> Path:
    """Download + unzip a Kaggle dataset into ``dest``.

    Skips the download if ``dest`` already contains images (unless ``force``).
    """
    import subprocess

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if not force and find_image_files(dest):
        LOG.info("Dataset already present in %s (skipping download).", dest)
        return dest

    if not ensure_kaggle_credentials():
        raise RuntimeError("Kaggle credentials unavailable; cannot download.")

    LOG.info("Downloading Kaggle dataset '%s' -> %s ...", slug, dest)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"kaggle download failed (exit {proc.returncode}).\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    n = len(find_image_files(dest))
    LOG.info("Download complete: %d image files under %s.", n, dest)
    if n == 0:
        raise RuntimeError(f"Download produced no images under {dest}.")
    return dest


# --------------------------------------------------------------------------- #
# Caching                                                                     #
# --------------------------------------------------------------------------- #
def cache_path(cache_dir: str | Path, smoke: bool = False) -> Path:
    name = "egypli_cache_smoke.pt" if smoke else "egypli_cache.pt"
    return Path(cache_dir) / name


def build_cache(
    disc: Discovery,
    cache_size: int,
    out_path: str | Path,
    subset_idx: Optional[Sequence[int]] = None,
) -> Path:
    """Decode every image once, resize to ``cache_size``, and save one uint8
    tensor + labels + metadata to ``out_path``.

    Parameters
    ----------
    subset_idx
        Optional indices into ``disc.paths`` to cache a subset (used for smoke).
    """
    import torch
    from PIL import Image

    idx = list(range(len(disc.paths))) if subset_idx is None else list(subset_idx)
    n = len(idx)
    arr = np.empty((n, 3, cache_size, cache_size), dtype=np.uint8)
    labels = np.empty((n,), dtype=np.int64)
    kept_paths: List[str] = []

    write = 0
    for src_i in idx:
        p = disc.paths[src_i]
        try:
            with Image.open(p) as im:
                im = im.convert("RGB").resize(
                    (cache_size, cache_size), Image.BILINEAR
                )
                a = np.asarray(im, dtype=np.uint8)          # [H, W, 3]
            arr[write] = np.transpose(a, (2, 0, 1))         # [3, H, W]
            labels[write] = disc.labels[src_i]
            kept_paths.append(str(p))
            write += 1
        except Exception as exc:  # noqa: BLE001 - skip corrupt, stay aligned
            LOG.warning("Skipping unreadable image %s (%s).", p, exc)

    arr = arr[:write]
    labels = labels[:write]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": torch.from_numpy(arr),          # uint8 [N, 3, S, S]
        "labels": torch.from_numpy(labels),       # int64 [N]
        "paths": kept_paths,
        "class_names": disc.class_names,
        "cache_size": cache_size,
    }
    torch.save(payload, out_path)
    nbytes = arr.nbytes
    LOG.info(
        "Cached %d images at %dx%d -> %s (%s).",
        write, cache_size, cache_size, out_path, bytes_to_human(nbytes),
    )
    return out_path


def load_cache(path: str | Path):
    """Load the cached payload dict (tensors on CPU)."""
    import torch

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cache not found: {path}. Run scripts/prepare_data.py first "
            f"(or notebook 01)."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- #
# Stratified splits                                                           #
# --------------------------------------------------------------------------- #
def make_stratified_splits(
    labels: np.ndarray,
    fractions: Tuple[float, float, float],
    seed: int,
) -> np.ndarray:
    """Return an array of split names ('train'/'val'/'test'), one per sample.

    Stratified: each class is split by the given fractions independently, using
    a fixed seed so all three models see identical data.
    """
    train_f, val_f, _ = fractions
    rng = np.random.default_rng(seed)
    split = np.empty(len(labels), dtype=object)
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(round(n * train_f))
        n_val = int(round(n * val_f))
        # Guard tiny classes: ensure at least 1 val + 1 test where possible.
        n_train = min(n_train, max(n - 2, 1)) if n >= 3 else max(n - 2, 0)
        n_val = min(n_val, n - n_train - 1) if n - n_train >= 2 else max(
            min(1, n - n_train), 0
        )
        split[cls_idx[:n_train]] = "train"
        split[cls_idx[n_train:n_train + n_val]] = "val"
        split[cls_idx[n_train + n_val:]] = "test"
    return split


def write_splits_csv(
    paths: Sequence[str],
    labels: np.ndarray,
    class_names: Sequence[str],
    split: np.ndarray,
    out_path: str | Path,
) -> Path:
    """Persist the split assignment to CSV for full reproducibility."""
    import csv

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "path", "label", "class_name", "split"])
        for i, (p, y, s) in enumerate(zip(paths, labels, split)):
            w.writerow([i, p, int(y), class_names[int(y)], s])
    counts = Counter(split.tolist())
    LOG.info(
        "Wrote splits -> %s (train=%d, val=%d, test=%d).",
        out_path, counts.get("train", 0), counts.get("val", 0), counts.get("test", 0),
    )
    return out_path


def read_splits_csv(path: str | Path) -> np.ndarray:
    """Read ``splits.csv`` and return the split-name array ordered by index.

    The returned array aligns positionally with the cached image tensor.
    """
    import csv

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Splits file not found: {path}. Run scripts/prepare_data.py first."
        )
    rows: List[Tuple[int, str]] = []
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append((int(r["index"]), r["split"]))
    rows.sort(key=lambda t: t[0])
    return np.array([s for _, s in rows], dtype=object)


def stratified_subset_indices(
    labels: np.ndarray, n_total: int, seed: int
) -> np.ndarray:
    """Pick ~``n_total`` indices, stratified by class (for the smoke run)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(1, n_total // len(classes))
    chosen: List[int] = []
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        take = min(per_class, len(cls_idx))
        chosen.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    return np.array(sorted(chosen), dtype=np.int64)


# --------------------------------------------------------------------------- #
# GPU-resident dataset                                                        #
# --------------------------------------------------------------------------- #
class GPUTensorDataset:
    """Holds the cached uint8 image tensor on-device and yields batches by
    indexing. No per-image decode, no DataLoader, no CPU->GPU copy per batch.
    """

    def __init__(self, images_uint8, labels, indices, device):
        import torch

        self.device = device
        self.images = images_uint8.to(device, non_blocking=True)  # uint8 [N,3,S,S]
        self.labels = labels.to(device, non_blocking=True).long()  # [N]
        self.indices = torch.as_tensor(
            np.asarray(indices), dtype=torch.long, device=device
        )

    def __len__(self) -> int:
        return int(self.indices.numel())

    def class_counts(self, num_classes: int):
        import torch

        y = self.labels[self.indices]
        return torch.bincount(y, minlength=num_classes)

    def loader(self, batch_size: int, shuffle: bool, seed: Optional[int] = None):
        """Yield (images_uint8 [B,3,S,S], labels [B]) batches, all on-device."""
        import torch

        idx = self.indices
        if shuffle:
            gen = torch.Generator()
            if seed is not None:
                gen.manual_seed(seed)
            perm = torch.randperm(idx.numel(), generator=gen).to(self.device)
            idx = idx[perm]
        for start in range(0, idx.numel(), batch_size):
            b = idx[start:start + batch_size]
            yield self.images[b], self.labels[b]


def splits_csv_path(metrics_dir: str | Path, smoke: bool = False) -> Path:
    name = "splits_smoke.csv" if smoke else "splits.csv"
    return Path(metrics_dir) / name


def load_split_datasets(cache: dict, split: np.ndarray, device):
    """Build train/val/test :class:`GPUTensorDataset` objects from a loaded
    cache and a split-name array."""
    images = cache["images"]
    labels = cache["labels"]
    out = {}
    for name in ("train", "val", "test"):
        idx = np.where(split == name)[0]
        out[name] = GPUTensorDataset(images, labels, idx, device)
    return out


def prepare_datasets(cfg, device, smoke: bool = False):
    """Load the cache + splits and build train/val/test GPU datasets.

    Returns ``(datasets, class_names)``. Raises a clear error if the cache and
    the splits file disagree (i.e. one was built with ``--smoke`` and the other
    without).
    """
    cache = load_cache(cache_path(cfg.paths.cache_dir, smoke))
    split = read_splits_csv(splits_csv_path(cfg.paths.metrics_dir, smoke))
    n_cache = int(cache["images"].shape[0])
    if len(split) != n_cache:
        raise ValueError(
            f"Split rows ({len(split)}) != cached images ({n_cache}). Rebuild "
            f"both with the SAME --smoke setting via scripts/prepare_data.py."
        )
    datasets = load_split_datasets(cache, split, device)
    return datasets, cache["class_names"]
