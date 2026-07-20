"""Convolutional Block Attention Module (CBAM).

Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

Two sequential attention branches applied as multiplicative gates:

1. **Channel attention** -- average- *and* max-pool the feature map over space,
   pass both through a *shared* MLP (reduction ratio 16), sum, sigmoid.
2. **Spatial attention** -- concatenate channel-wise average and max maps, run a
   single 7x7 convolution, sigmoid.

Identity initialisation
-----------------------
Each gate is applied as a residual gate ``x * (1 + gamma * (M - 1))`` with a
learnable scalar ``gamma`` initialised to **0**. At the first step the module is
therefore an exact identity, so the pretrained ConvNeXt features are not
destroyed; the network learns how much attention to apply as ``gamma`` moves away
from zero.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention via shared MLP over avg- and max-pooled descriptors."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        # 1x1 convs implement the shared MLP on [B, C, 1, 1] descriptors.
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return torch.sigmoid(avg + mx)  # [B, C, 1, 1]


class SpatialAttention(nn.Module):
    """Spatial attention via a 7x7 conv over channel-wise avg/max maps."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "spatial kernel must be odd"
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        concat = torch.cat([avg, mx], dim=1)  # [B, 2, H, W]
        return torch.sigmoid(self.conv(concat))  # [B, 1, H, W]


class CBAM(nn.Module):
    """Full CBAM block with identity-initialised residual gates.

    Parameters
    ----------
    channels
        Number of input feature-map channels.
    reduction
        Channel-attention MLP reduction ratio (default 16).
    kernel_size
        Spatial-attention convolution size (default 7).
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)
        # gammas start at 0 -> module is identity at init.
        self.gamma_c = nn.Parameter(torch.zeros(1))
        self.gamma_s = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mc = self.channel(x)
        x = x * (1.0 + self.gamma_c * (mc - 1.0))
        ms = self.spatial(x)
        x = x * (1.0 + self.gamma_s * (ms - 1.0))
        return x


def _self_test() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 64, 8, 8)
    cbam = CBAM(64)
    y = cbam(x)
    assert y.shape == x.shape, y.shape
    # Identity at initialisation (gammas == 0).
    assert torch.allclose(y, x, atol=1e-6), "CBAM should be identity at init"
    # After nudging gammas it should change the output.
    with torch.no_grad():
        cbam.gamma_c.fill_(0.5)
        cbam.gamma_s.fill_(0.5)
    y2 = cbam(x)
    assert not torch.allclose(y2, x), "CBAM should modify features once active"
    print("cbam.py self-test passed: shape preserved, identity at init, active after.")


if __name__ == "__main__":
    _self_test()
