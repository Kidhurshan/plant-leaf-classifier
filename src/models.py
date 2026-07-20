"""Model builders: three timm backbones + one identical shared head.

The head is byte-for-byte identical across all three models so the only variable
in the comparison is the backbone (the controlled-experiment requirement). The
proposed model wraps ConvNeXt-Tiny with CBAM on its later stages.

All backbones are created with ``num_classes=0, global_pool=''`` so we obtain a
feature *map*; the shared head does its own global average pooling. Feature-map
layout is normalised to NCHW to transparently support CNNs (NCHW), Swin (NHWC)
and any token-sequence output (NLC).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn

from src.cbam import CBAM
from src.config import Config
from src.utils import LOG, count_parameters, format_param_count


# --------------------------------------------------------------------------- #
# Shared head                                                                 #
# --------------------------------------------------------------------------- #
class SharedHead(nn.Module):
    """Global average pool -> LayerNorm -> Dropout -> Linear(num_classes).

    Identical across all three models.
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        x = feature_map.mean(dim=(2, 3))  # global average pooling -> [B, C]
        x = self.norm(x)
        x = self.drop(x)
        return self.fc(x)


# --------------------------------------------------------------------------- #
# Backbone name resolution                                                    #
# --------------------------------------------------------------------------- #
def _pretrained_names() -> set:
    return set(timm.list_models(pretrained=True))


def resolve_backbone(primary: str, fallback: str) -> str:
    """Return ``primary`` if it exists as a pretrained timm model, else
    ``fallback`` (with a clear printed message). Raises if neither exists."""
    available = _pretrained_names()
    if primary in available:
        return primary
    LOG.warning(
        "Primary backbone '%s' not found in this timm build; "
        "falling back to '%s'.", primary, fallback,
    )
    print(f"[models] '{primary}' unavailable -> using fallback '{fallback}'.")
    if fallback in available:
        return fallback
    raise ValueError(
        f"Neither primary '{primary}' nor fallback '{fallback}' is available in "
        f"timm {timm.__version__}. Update timm or pick another checkpoint."
    )


# --------------------------------------------------------------------------- #
# Classifier                                                                  #
# --------------------------------------------------------------------------- #
class LeafClassifier(nn.Module):
    """A timm backbone (optionally CBAM-wrapped) + the shared head."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        dropout: float,
        use_cbam: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.use_cbam = use_cbam
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool=""
        )
        self.num_features: int = int(self.backbone.num_features)

        # CBAM modules are stored at the top level (NOT inside .backbone) so that
        # freezing the backbone in phase 1 leaves them trainable.
        self.cbam_modules = nn.ModuleList()
        self._hooks: List = []
        self._cbam_final: Optional[CBAM] = None
        if use_cbam:
            self._attach_cbam()

        self.head = SharedHead(self.num_features, num_classes, dropout)

    # -- CBAM insertion ----------------------------------------------------- #
    def _make_hook(self, cbam: CBAM):
        def hook(_module, _inp, output):
            # ConvNeXt stage outputs are NCHW; replace with the gated output.
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                return cbam(output)
            return output
        return hook

    def _attach_cbam(self) -> None:
        """Attach CBAM after the last two ConvNeXt stages via forward hooks.

        Falls back to a single CBAM on the final feature map if the stage
        structure is not accessible.
        """
        stages = getattr(self.backbone, "stages", None)
        attached = False
        if stages is not None:
            try:
                stage_list = list(stages)
                channels = list(self.backbone.feature_info.channels())
                targets = [len(stage_list) - 2, len(stage_list) - 1]
                for si in targets:
                    ch = channels[si]
                    cbam = CBAM(ch)
                    self.cbam_modules.append(cbam)
                    self._hooks.append(
                        stage_list[si].register_forward_hook(self._make_hook(cbam))
                    )
                attached = True
                LOG.info("CBAM attached to ConvNeXt stages %s (channels %s).",
                         targets, [channels[i] for i in targets])
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Stage-hook CBAM failed (%s); using final-map CBAM.", exc)
        if not attached:
            self._cbam_final = CBAM(self.num_features)
            self.cbam_modules.append(self._cbam_final)
            LOG.info("CBAM attached to final feature map (%d ch).", self.num_features)

    # -- Feature extraction ------------------------------------------------- #
    def _to_nchw(self, feats: torch.Tensor) -> torch.Tensor:
        """Normalise backbone feature output to NCHW."""
        c = self.num_features
        if feats.ndim == 4:
            if feats.shape[1] == c:
                return feats                                   # NCHW
            if feats.shape[-1] == c:
                return feats.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
            return feats                                       # assume NCHW
        if feats.ndim == 3:
            b, d1, d2 = feats.shape
            if d2 == c:                                        # [B, L, C]
                s = int(round(d1 ** 0.5))
                return feats.transpose(1, 2).reshape(b, c, s, s)
            if d1 == c:                                        # [B, C, L]
                s = int(round(d2 ** 0.5))
                return feats.reshape(b, c, s, s)
        if feats.ndim == 2:                                    # already pooled
            return feats[:, :, None, None]
        raise RuntimeError(
            f"Unexpected feature shape {tuple(feats.shape)} for C={c}."
        )

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        feats = self._to_nchw(feats)
        if self._cbam_final is not None:
            feats = self._cbam_final(feats)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_feature_map(x))

    # -- Freeze / unfreeze -------------------------------------------------- #
    def set_backbone_trainable(self, trainable: bool) -> None:
        """Phase 1 freezes the pretrained backbone (head + CBAM stay trainable);
        phase 2 unfreezes everything."""
        for p in self.backbone.parameters():
            p.requires_grad_(trainable)

    def trainable_summary(self) -> Dict[str, int]:
        return count_parameters(self)


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
def build_model(cfg: Config, model_key: str, device=None) -> LeafClassifier:
    """Build one of the three configured models and move it to ``device``."""
    mdef = cfg.model_def(model_key)
    name = resolve_backbone(mdef.backbone, mdef.fallback)
    model = LeafClassifier(
        backbone_name=name,
        num_classes=cfg.data.num_classes,
        dropout=cfg.head.dropout,
        use_cbam=mdef.cbam,
    )
    if device is not None:
        model = model.to(device)
    counts = model.trainable_summary()
    LOG.info(
        "Built '%s' (backbone=%s, cbam=%s): %s params (%s trainable).",
        model_key, name, mdef.cbam,
        format_param_count(counts["total"]),
        format_param_count(counts["trainable"]),
    )
    return model


def load_trained_model(cfg: Config, model_key: str, ckpt_path, device=None) -> LeafClassifier:
    """Build the model and load weights from a saved checkpoint.

    Used by ``scripts/evaluate_all.py`` and ``scripts/predict.py`` so evaluation
    and the live demo depend only on saved checkpoints, never on training state.
    """
    from pathlib import Path
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train '{model_key}' first "
            f"(scripts/train.py --model {model_key})."
        )
    model = build_model(cfg, model_key, device=device)
    ckpt = torch.load(ckpt_path, map_location=device or "cpu", weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    LOG.info("Loaded '%s' weights from %s (val macro-F1=%.4f).",
             model_key, ckpt_path, ckpt.get("val_macro_f1", float("nan")))
    return model


def describe_model(model: LeafClassifier) -> str:
    """Short human-readable summary: backbone, feature dim, param counts, and
    the top-level child modules."""
    counts = model.trainable_summary()
    lines = [
        f"Backbone      : {model.backbone_name}",
        f"CBAM          : {model.use_cbam} "
        f"({len(model.cbam_modules)} module(s))",
        f"Feature dim   : {model.num_features}",
        f"Total params  : {counts['total']:,} ({format_param_count(counts['total'])})",
        f"Trainable     : {counts['trainable']:,} "
        f"({format_param_count(counts['trainable'])})",
        "Top-level modules:",
    ]
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        lines.append(f"  - {name:<14} {child.__class__.__name__:<20} "
                     f"{format_param_count(n):>8} params")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Shape assertions (used by notebooks / tests)                                #
# --------------------------------------------------------------------------- #
def assert_model_shapes(
    cfg: Config, device=None, batch: int = 2, verbose: bool = True
) -> None:
    """Build all three models, run a dummy forward, and assert output shape
    is ``[batch, num_classes]`` for each."""
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    x = torch.randn(batch, 3, cfg.data.img_size, cfg.data.img_size, device=device)
    for key in cfg.model_list:
        model = build_model(cfg, key, device=device).eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch, cfg.data.num_classes), (
            f"{key}: expected {(batch, cfg.data.num_classes)}, got {tuple(out.shape)}"
        )
        if verbose:
            print(f"  [OK] {key:<16} output {tuple(out.shape)}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"All {len(cfg.model_list)} models produce "
          f"[{batch}, {cfg.data.num_classes}] logits. Shape assertions passed.")


if __name__ == "__main__":  # `python -m src.models`  (downloads pretrained weights)
    from src.config import load_config
    assert_model_shapes(load_config())
