"""GPU batch augmentation.

Everything runs on-device on a whole uint8 batch -- no per-image CPU decode, no
DataLoader. Geometric augmentation (random-resized-crop + rotation + translation)
is done in a single ``affine_grid`` / ``grid_sample`` call with **per-sample**
parameters; flips, colour jitter and random erasing are batched tensor ops.

Mixup / CutMix live here too and are applied by the engine in phase 2 only.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

# ImageNet statistics (all three backbones are ImageNet-pretrained).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Rec.601 luma weights for saturation/grayscale.
_LUMA = (0.299, 0.587, 0.114)


class GPUAugment:
    """Batched GPU augmentation pipeline.

    Call the instance with a uint8 batch ``[B, 3, S, S]`` and get a normalised
    float batch ``[B, 3, img_size, img_size]``. Training mode applies the full
    stochastic pipeline; eval mode is a deterministic centre-crop + normalise.
    """

    def __init__(self, aug_cfg, img_size: int, device, training: bool = True):
        self.cfg = aug_cfg
        self.img_size = img_size
        self.device = device
        self.training = training
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.luma = torch.tensor(_LUMA, device=device).view(1, 3, 1, 1)

    # -- helpers ------------------------------------------------------------ #
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _center_crop(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        s = self.img_size
        if h == s and w == s:
            return x
        if h < s or w < s:  # upscale first if cache smaller than target
            x = F.interpolate(x, size=(max(h, s), max(w, s)),
                              mode="bilinear", align_corners=False)
            _, _, h, w = x.shape
        top = (h - s) // 2
        left = (w - s) // 2
        return x[:, :, top:top + s, left:left + s]

    def _geometric(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample random-resized-crop + rotation + translation via an affine
        grid sampled at the output resolution."""
        b = x.shape[0]
        dev = x.device
        lo, hi = self.cfg.rrc_scale
        area = torch.empty(b, device=dev).uniform_(lo, hi)
        scale = area.sqrt()                                   # linear region size
        max_deg = self.cfg.rotation_deg
        ang = torch.empty(b, device=dev).uniform_(-max_deg, max_deg) * math.pi / 180.0
        cos, sin = torch.cos(ang), torch.sin(ang)
        # Rotation matrix scaled by the crop factor.
        a00 = cos * scale
        a01 = -sin * scale
        a10 = sin * scale
        a11 = cos * scale
        # Translation kept inside the image so the crop stays valid.
        room = (1.0 - scale).clamp(min=0.0)
        tx = (torch.empty(b, device=dev).uniform_(-1, 1)) * room
        ty = (torch.empty(b, device=dev).uniform_(-1, 1)) * room
        theta = torch.stack(
            [torch.stack([a00, a01, tx], dim=1),
             torch.stack([a10, a11, ty], dim=1)],
            dim=1,
        )  # [B, 2, 3]
        grid = F.affine_grid(
            theta, (b, x.shape[1], self.img_size, self.img_size),
            align_corners=False,
        )
        return F.grid_sample(
            x, grid, mode="bilinear", padding_mode="reflection", align_corners=False
        )

    def _flip(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        if self.cfg.hflip:
            m = (torch.rand(b, device=x.device) < 0.5).view(b, 1, 1, 1)
            x = torch.where(m, x.flip(-1), x)
        if self.cfg.vflip:
            m = (torch.rand(b, device=x.device) < 0.5).view(b, 1, 1, 1)
            x = torch.where(m, x.flip(-2), x)
        return x

    def _color_jitter(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        cj = self.cfg.color_jitter

        def factor(strength: float) -> torch.Tensor:
            return torch.empty(b, 1, 1, 1, device=x.device).uniform_(
                max(0.0, 1.0 - strength), 1.0 + strength
            )

        if cj.brightness > 0:
            x = x * factor(cj.brightness)
        if cj.contrast > 0:
            gray = (x * self.luma).sum(dim=1, keepdim=True)
            mean = gray.mean(dim=(2, 3), keepdim=True)
            x = (x - mean) * factor(cj.contrast) + mean
        if cj.saturation > 0:
            gray = (x * self.luma).sum(dim=1, keepdim=True)
            x = gray + (x - gray) * factor(cj.saturation)
        return x.clamp(0.0, 1.0)

    def _random_erasing(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample random erasing (operates in normalised space; erased region
        is set to 0 == the dataset mean)."""
        p = self.cfg.random_erasing_p
        if p <= 0:
            return x
        b, _, h, w = x.shape
        do = torch.rand(b, device=x.device) < p
        idx = torch.nonzero(do, as_tuple=False).flatten().tolist()
        for i in idx:
            area = float(torch.empty(1).uniform_(0.02, 0.2)) * h * w
            aspect = float(torch.empty(1).uniform_(0.3, 3.3))
            eh = int(round(math.sqrt(area * aspect)))
            ew = int(round(math.sqrt(area / aspect)))
            if eh < 1 or ew < 1 or eh >= h or ew >= w:
                continue
            top = int(torch.randint(0, h - eh, (1,)))
            left = int(torch.randint(0, w - ew, (1,)))
            x[i, :, top:top + eh, left:left + ew] = 0.0
        return x

    # -- entry point -------------------------------------------------------- #
    def __call__(self, x_uint8: torch.Tensor) -> torch.Tensor:
        x = x_uint8.to(self.device, non_blocking=True).float().div_(255.0)
        if not self.training:
            return self._normalize(self._center_crop(x))
        x = self._geometric(x)
        x = self._flip(x)
        x = self._color_jitter(x)
        x = self._normalize(x)
        x = self._random_erasing(x)
        return x


# --------------------------------------------------------------------------- #
# Mixup / CutMix                                                              #
# --------------------------------------------------------------------------- #
def _rand_bbox(h: int, w: int, lam: float) -> Tuple[int, int, int, int]:
    cut = math.sqrt(1.0 - lam)
    cw, ch = int(w * cut), int(h * cut)
    cx, cy = int(torch.randint(0, w, (1,))), int(torch.randint(0, h, (1,)))
    x1 = max(cx - cw // 2, 0)
    y1 = max(cy - ch // 2, 0)
    x2 = min(cx + cw // 2, w)
    y2 = min(cy + ch // 2, h)
    return x1, y1, x2, y2


def apply_mix(x: torch.Tensor, y: torch.Tensor, aug_cfg):
    """Maybe apply mixup or cutmix to a normalised batch.

    Returns ``(x, y_a, y_b, lam)``. When no mixing is applied, ``y_a == y_b`` and
    ``lam == 1`` so the engine's ``lam*loss(y_a) + (1-lam)*loss(y_b)`` reduces to
    the plain loss.
    """
    if not (aug_cfg.mixup or aug_cfg.cutmix):
        return x, y, y, 1.0
    if torch.rand(1).item() >= aug_cfg.mix_prob:
        return x, y, y, 1.0

    b, _, h, w = x.shape
    perm = torch.randperm(b, device=x.device)
    use_cutmix = aug_cfg.cutmix and (not aug_cfg.mixup or torch.rand(1).item() < 0.5)

    if use_cutmix:
        lam = float(torch.distributions.Beta(
            aug_cfg.cutmix_alpha, aug_cfg.cutmix_alpha).sample())
        x1, y1, x2, y2 = _rand_bbox(h, w, lam)
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (h * w))  # correct for actual area
    else:
        lam = float(torch.distributions.Beta(
            aug_cfg.mixup_alpha, aug_cfg.mixup_alpha).sample())
        x = lam * x + (1.0 - lam) * x[perm]

    return x, y, y[perm], lam


def denormalize(x: torch.Tensor, device=None) -> torch.Tensor:
    """Invert ImageNet normalisation to [0, 1] for visualisation."""
    dev = device or x.device
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


if __name__ == "__main__":
    from types import SimpleNamespace

    cj = SimpleNamespace(brightness=0.3, contrast=0.3, saturation=0.3)
    aug_cfg = SimpleNamespace(
        rrc_scale=[0.7, 1.0], hflip=True, vflip=True, rotation_deg=30,
        color_jitter=cj, random_erasing_p=0.25, mixup=True, cutmix=True,
        mixup_alpha=0.2, cutmix_alpha=1.0, mix_prob=1.0,
    )
    x = (torch.rand(4, 3, 256, 256) * 255).to(torch.uint8)
    aug = GPUAugment(aug_cfg, img_size=224, device=torch.device("cpu"))
    out = aug(x)
    assert out.shape == (4, 3, 224, 224), out.shape
    y = torch.randint(0, 8, (4,))
    xn, ya, yb, lam = apply_mix(out, y, aug_cfg)
    assert xn.shape == out.shape and 0.0 <= lam <= 1.0
    print(f"augment.py self-test passed: out {tuple(out.shape)}, lam={lam:.3f}")
