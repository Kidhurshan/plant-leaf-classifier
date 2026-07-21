"""YAML loading, typed dataclass config, and validation.

The whole project is configured from a single YAML file (``configs/default.yaml``
by default). Loading validates the config and fails early with a clear message
rather than blowing up deep inside training.

Usage
-----
>>> from src.config import load_config
>>> cfg = load_config("configs/default.yaml")
>>> cfg.train.batch_size
64
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List

import yaml


# --------------------------------------------------------------------------- #
# Typed sub-configs                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class Paths:
    data_dir: str
    cache_dir: str
    checkpoint_dir: str
    results_dir: str
    figures_dir: str
    metrics_dir: str
    gradcam_dir: str

    def as_paths(self) -> Dict[str, Path]:
        return {k: Path(v) for k, v in asdict(self).items()}

    def ensure_dirs(self) -> None:
        """Create every output directory (idempotent)."""
        for name, p in self.as_paths().items():
            if name == "data_dir":
                continue  # created by the downloader
            Path(p).mkdir(parents=True, exist_ok=True)


@dataclass
class Split:
    train: float
    val: float
    test: float


@dataclass
class DataCfg:
    kaggle_slug: str
    num_classes: int
    expected_total: int
    cache_size: int
    img_size: int
    split: Split
    group_aware_split: bool = True


@dataclass
class ModelDef:
    backbone: str
    fallback: str
    cbam: bool = False


@dataclass
class HeadCfg:
    dropout: float


@dataclass
class TrainCfg:
    batch_size: int
    phase1_epochs: int
    phase2_epochs: int
    phase1_lr: float
    phase2_lr: float
    warmup_epochs: int
    weight_decay: float
    grad_clip: float
    patience: int
    label_smoothing: float
    use_llrd: bool
    llrd_decay: float


@dataclass
class ColorJitter:
    brightness: float
    contrast: float
    saturation: float


@dataclass
class AugmentCfg:
    rrc_scale: List[float]
    hflip: bool
    vflip: bool
    rotation_deg: float
    color_jitter: ColorJitter
    random_erasing_p: float
    mixup: bool
    cutmix: bool
    mixup_alpha: float
    cutmix_alpha: float
    mix_prob: float
    mix_phase2_only: bool


@dataclass
class LossCfg:
    name: str
    focal_gamma: float
    focal_alpha: str
    label_smoothing: float


@dataclass
class EvalCfg:
    tta: bool


@dataclass
class SmokeCfg:
    n_images: int
    epochs: int
    batch_size: int


@dataclass
class Config:
    seed: int
    paths: Paths
    data: DataCfg
    models: Dict[str, ModelDef]
    model_list: List[str]
    head: HeadCfg
    train: TrainCfg
    augment: AugmentCfg
    loss: LossCfg
    eval: EvalCfg
    smoke: SmokeCfg
    # The raw dict, kept verbatim for run-metadata JSON (full traceability).
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # -- convenience -------------------------------------------------------- #
    def model_def(self, key: str) -> ModelDef:
        if key not in self.models:
            raise KeyError(
                f"Unknown model key '{key}'. Known: {list(self.models)}"
            )
        return self.models[key]

    def to_dict(self) -> Dict[str, Any]:
        """Full config as a plain dict (for run-meta JSON)."""
        return self.raw


# --------------------------------------------------------------------------- #
# Loading + validation                                                        #
# --------------------------------------------------------------------------- #
class ConfigError(ValueError):
    """Raised when the config is missing keys or has invalid values."""


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required config key '{key}' in section '{ctx}'.")
    return d[key]


def _build(raw: Dict[str, Any]) -> Config:
    paths = Paths(**_require(raw, "paths", "root"))

    data_raw = dict(_require(raw, "data", "root"))
    split = Split(**_require(data_raw, "split", "data"))
    data_raw["split"] = split
    data = DataCfg(**data_raw)

    models_raw = _require(raw, "models", "root")
    models = {k: ModelDef(**v) for k, v in models_raw.items()}

    head = HeadCfg(**_require(raw, "head", "root"))
    train = TrainCfg(**_require(raw, "train", "root"))

    aug_raw = dict(_require(raw, "augment", "root"))
    aug_raw["color_jitter"] = ColorJitter(**aug_raw["color_jitter"])
    augment = AugmentCfg(**aug_raw)

    loss = LossCfg(**_require(raw, "loss", "root"))
    eval_cfg = EvalCfg(**_require(raw, "eval", "root"))
    smoke = SmokeCfg(**_require(raw, "smoke", "root"))

    return Config(
        seed=int(_require(raw, "seed", "root")),
        paths=paths,
        data=data,
        models=models,
        model_list=list(_require(raw, "model_list", "root")),
        head=head,
        train=train,
        augment=augment,
        loss=loss,
        eval=eval_cfg,
        smoke=smoke,
        raw=raw,
    )


def _validate(cfg: Config) -> None:
    errs: List[str] = []

    if cfg.data.num_classes < 2:
        errs.append(f"data.num_classes must be >= 2, got {cfg.data.num_classes}")
    if cfg.data.img_size <= 0 or cfg.data.cache_size < cfg.data.img_size:
        errs.append(
            f"data.cache_size ({cfg.data.cache_size}) must be >= img_size "
            f"({cfg.data.img_size}) and positive"
        )

    s = cfg.data.split
    total = s.train + s.val + s.test
    if abs(total - 1.0) > 1e-6:
        errs.append(f"data.split fractions must sum to 1.0, got {total:.4f}")
    if min(s.train, s.val, s.test) <= 0:
        errs.append("data.split fractions must all be > 0")

    if not cfg.model_list:
        errs.append("model_list is empty")
    for key in cfg.model_list:
        if key not in cfg.models:
            errs.append(f"model_list references unknown model '{key}'")

    if cfg.loss.name not in {"focal", "ce"}:
        errs.append(f"loss.name must be 'focal' or 'ce', got '{cfg.loss.name}'")
    if cfg.loss.focal_alpha not in {"inverse_freq", "none"}:
        errs.append(
            f"loss.focal_alpha must be 'inverse_freq' or 'none', "
            f"got '{cfg.loss.focal_alpha}'"
        )

    if not (0.0 <= cfg.head.dropout < 1.0):
        errs.append(f"head.dropout must be in [0, 1), got {cfg.head.dropout}")
    if cfg.train.batch_size <= 0:
        errs.append("train.batch_size must be > 0")
    if cfg.train.phase1_epochs < 0 or cfg.train.phase2_epochs < 1:
        errs.append("train.phase1_epochs >= 0 and phase2_epochs >= 1 required")
    if cfg.train.patience < 1:
        errs.append("train.patience must be >= 1")
    if not (0.0 <= cfg.augment.random_erasing_p <= 1.0):
        errs.append("augment.random_erasing_p must be in [0, 1]")
    if len(cfg.augment.rrc_scale) != 2 or cfg.augment.rrc_scale[0] > cfg.augment.rrc_scale[1]:
        errs.append("augment.rrc_scale must be [low, high] with low <= high")

    if errs:
        bullet = "\n  - ".join(errs)
        raise ConfigError(f"Invalid config ({len(errs)} problem(s)):\n  - {bullet}")


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load, build and validate the project config from a YAML file.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ConfigError
        If a required key is missing or a value is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path.resolve()}. "
            f"Pass a valid --config path (default: configs/default.yaml)."
        )
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")

    cfg = _build(raw)
    _validate(cfg)
    return cfg


if __name__ == "__main__":  # quick self-check: `python -m src.config`
    c = load_config()
    print("Config OK.")
    print(f"  seed          = {c.seed}")
    print(f"  models        = {c.model_list}")
    print(f"  img_size      = {c.data.img_size} (cache {c.data.cache_size})")
    print(f"  loss          = {c.loss.name}")
    print(f"  batch_size    = {c.train.batch_size}")
