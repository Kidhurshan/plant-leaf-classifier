"""Loss functions: focal loss (default) and cross-entropy with label smoothing.

Focal loss addresses class imbalance (under-represented tomato/persimmon) with a
per-class ``alpha`` derived from inverse class frequency and a focusing term
``(1 - p_t) ** gamma``. The alternative is standard cross-entropy with label
smoothing. Both take hard integer targets; mixup/cutmix is handled at the engine
level by combining two hard-target losses.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_frequency_alpha(class_counts: torch.Tensor) -> torch.Tensor:
    """Per-class weights ~ inverse frequency, normalised to mean 1.

    ``w_c = N / (K * count_c)`` then rescaled so ``mean(w) == 1`` (keeps the loss
    scale comparable to the unweighted case).
    """
    counts = class_counts.float().clamp(min=1.0)
    n = counts.sum()
    k = counts.numel()
    w = n / (k * counts)
    return w / w.mean()


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weighting.

    Parameters
    ----------
    gamma
        Focusing parameter (0 == weighted cross-entropy).
    alpha
        Optional per-class weight tensor of shape ``[num_classes]``.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        # Per-sample CE (alpha applied via nll_loss weight if present).
        ce = F.nll_loss(log_p, target, weight=self.alpha, reduction="none")
        p_t = log_p.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        loss = (1.0 - p_t) ** self.gamma * ce
        return loss.mean()


def build_loss(cfg, class_counts: torch.Tensor, device) -> nn.Module:
    """Construct the configured criterion.

    Parameters
    ----------
    cfg
        The project :class:`~src.config.Config`.
    class_counts
        Per-class sample counts from the **training** split (for focal alpha).
    device
        Target device for any weight tensors.
    """
    loss_cfg = cfg.loss
    if loss_cfg.name == "focal":
        alpha = None
        if loss_cfg.focal_alpha == "inverse_freq":
            alpha = inverse_frequency_alpha(class_counts).to(device)
        return FocalLoss(gamma=loss_cfg.focal_gamma, alpha=alpha)
    if loss_cfg.name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=loss_cfg.label_smoothing)
    raise ValueError(f"Unknown loss '{loss_cfg.name}'")


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(8, 8, requires_grad=True)
    target = torch.randint(0, 8, (8,))
    counts = torch.tensor([100, 90, 80, 70, 60, 50, 20, 10])
    fl = FocalLoss(2.0, inverse_frequency_alpha(counts))
    loss = fl(logits, target)
    loss.backward()
    assert torch.isfinite(loss), loss
    print(f"losses.py self-test passed: focal loss = {loss.item():.4f}")
