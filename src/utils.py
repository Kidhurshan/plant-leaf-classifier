"""Cross-cutting utilities: seeding, device/AMP detection, git sync, timing,
logging, and run-metadata recording.

These functions are deliberately dependency-light so they can be called from the
very first notebook cell as well as from every script.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str = "task4", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that prints cleanly in notebooks."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


LOG = get_logger()


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch and CUDA from a single value.

    Parameters
    ----------
    seed
        The master seed.
    deterministic
        If True, force deterministic cuDNN algorithms. This is slower and is
        left off by default; the fixed seed already makes splits reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------- #
# Device / AMP detection                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class AmpConfig:
    """Resolved automatic-mixed-precision settings for the current device."""

    device: torch.device
    use_amp: bool           # whether to wrap forward passes in autocast
    amp_dtype: torch.dtype  # bfloat16 where supported, else float16
    needs_scaler: bool      # GradScaler is only required for float16

    @property
    def dtype_name(self) -> str:
        return str(self.amp_dtype).replace("torch.", "")


def get_device() -> torch.device:
    """Return the best available device (CUDA if present, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_amp(device: Optional[torch.device] = None) -> AmpConfig:
    """Detect AMP capability at runtime.

    bfloat16 is preferred where the GPU supports it (A100, L4, ...), otherwise
    float16 with a GradScaler. On CPU, AMP is disabled.
    """
    device = device or get_device()
    if device.type != "cuda":
        return AmpConfig(device=device, use_amp=False,
                         amp_dtype=torch.float32, needs_scaler=False)
    bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    return AmpConfig(
        device=device,
        use_amp=True,
        amp_dtype=amp_dtype,
        needs_scaler=(amp_dtype == torch.float16),
    )


def gpu_report() -> Dict[str, Any]:
    """Print and return a concise environment / GPU report."""
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    print("=" * 60)
    print("ENVIRONMENT")
    print("=" * 60)
    print(f"  Python        : {info['python']}")
    print(f"  PyTorch       : {info['torch']}")
    print(f"  CUDA available: {info['cuda_available']}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        bf16 = torch.cuda.is_bf16_supported()
        info.update(gpu=name, vram_gb=round(total_gb, 1),
                    cuda_version=torch.version.cuda, bf16_supported=bf16)
        print(f"  GPU           : {name}")
        print(f"  VRAM          : {total_gb:.1f} GB")
        print(f"  CUDA runtime  : {torch.version.cuda}")
        print(f"  bfloat16      : {'YES' if bf16 else 'no (will use float16)'}")
    else:
        print("  GPU           : NONE (CPU only -- training will be very slow)")
    print("=" * 60)
    return info


# --------------------------------------------------------------------------- #
# Git / sync helpers                                                          #
# --------------------------------------------------------------------------- #
def _git(args: list[str], cwd: Optional[Path] = None) -> str:
    """Run a git command and return stripped stdout ('' on failure)."""
    try:
        out = subprocess.check_output(
            ["git", *args], cwd=str(cwd) if cwd else None,
            stderr=subprocess.STDOUT, text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def get_git_commit(short: bool = True, cwd: Optional[Path] = None) -> str:
    """Return the current git commit hash, or 'unknown' outside a repo."""
    args = ["rev-parse", "--short", "HEAD"] if short else ["rev-parse", "HEAD"]
    commit = _git(args, cwd=cwd)
    return commit or "unknown"


def git_is_dirty(cwd: Optional[Path] = None) -> bool:
    """True if the working tree has uncommitted changes."""
    return bool(_git(["status", "--porcelain"], cwd=cwd))


def sync_repo(cwd: Optional[Path] = None) -> str:
    """`git pull` then print + return the current commit hash.

    Call this at the top of every training notebook after editing ``src/``
    locally and pushing. With ``%autoreload`` on, changes take effect without a
    kernel restart.
    """
    print("Syncing repository (git pull)...")
    pull = _git(["pull", "--ff-only"], cwd=cwd)
    print(pull or "(git pull produced no output)")
    commit = get_git_commit(short=True, cwd=cwd)
    dirty = " [DIRTY WORKING TREE]" if git_is_dirty(cwd) else ""
    print(f"Now running code at commit: {commit}{dirty}")
    if pull and "up to date" not in pull.lower():
        print(
            "\n" + "!" * 68 +
            "\n!! The pull brought NEW CODE into this runtime."
            "\n!! %autoreload CANNOT reload changed classes/dataclasses."
            "\n!! -> RESTART THE KERNEL now, then run the notebook from the top."
            "\n" + "!" * 68
        )
    return commit


# --------------------------------------------------------------------------- #
# Timing                                                                       #
# --------------------------------------------------------------------------- #
def human_time(seconds: float) -> str:
    """Format a duration in seconds as e.g. '1h 02m 03s' or '45.2s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


class Timer:
    """Simple stopwatch. ``t = Timer(); ...; t.stop()`` -> elapsed seconds."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.elapsed = 0.0

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self.start
        return self.elapsed


@contextmanager
def time_block(label: str, logger: Optional[logging.Logger] = None):
    """Context manager that logs how long a block took."""
    logger = logger or LOG
    t0 = time.perf_counter()
    yield
    logger.info("%s took %s", label, human_time(time.perf_counter() - t0))


# --------------------------------------------------------------------------- #
# Model helpers                                                               #
# --------------------------------------------------------------------------- #
def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def format_param_count(n: int) -> str:
    """Human-readable parameter count, e.g. 21_400_000 -> '21.4M'."""
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


# --------------------------------------------------------------------------- #
# Time / metadata                                                             #
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pkg_version(name: str) -> str:
    try:
        module = __import__(name)
        return getattr(module, "__version__", "unknown")
    except Exception:  # noqa: BLE001 - best-effort version probe
        return "not-installed"


def collect_env_meta() -> Dict[str, Any]:
    """Library versions, GPU name, CUDA version and git commit hash."""
    meta: Dict[str, Any] = {
        "timestamp_utc": now_iso(),
        "git_commit": get_git_commit(short=False),
        "git_commit_short": get_git_commit(short=True),
        "git_dirty": git_is_dirty(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _pkg_version("torch"),
        "torchvision": _pkg_version("torchvision"),
        "timm": _pkg_version("timm"),
        "numpy": _pkg_version("numpy"),
        "sklearn": _pkg_version("sklearn"),
    }
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        meta["gpu"] = torch.cuda.get_device_name(idx)
        meta["cuda"] = torch.version.cuda
        meta["bf16_supported"] = torch.cuda.is_bf16_supported()
    else:
        meta["gpu"] = "cpu"
    return meta


def save_run_meta(
    path: str | Path,
    *,
    model_key: str,
    seed: int,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``{model}_run_meta.json`` capturing seed, versions, GPU, commit
    hash and the full config for complete reproducibility/traceability."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model_key,
        "seed": seed,
        "env": collect_env_meta(),
        "config": config,
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


# --------------------------------------------------------------------------- #
# Misc                                                                         #
# --------------------------------------------------------------------------- #
def file_fingerprint(path: str | Path) -> str:
    """Short md5 of a file's contents ('missing' if absent).

    Used to stamp checkpoints with the data split they were trained on, so a
    changed split can never be silently resumed into.
    """
    import hashlib

    path = Path(path)
    if not path.exists():
        return "missing"
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def bytes_to_human(n: int) -> str:
    """Bytes -> human string, e.g. 705_000_000 -> '672.3 MB'."""
    step = 1024.0
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < step:
            return f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} PB"
