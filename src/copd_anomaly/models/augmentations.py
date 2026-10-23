"""
3D Volumetric Data Augmentations for Contrastive Pretraining.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Random3DFlip(nn.Module):
    """Randomly flips 3D patches across spatial dimensions."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial dims for (C, D, H, W) are 1, 2, 3
        # Spatial dims for (B, C, D, H, W) are 2, 3, 4
        is_batched = x.ndim == 5
        spatial_dims = [2, 3, 4] if is_batched else [1, 2, 3]

        for dim in spatial_dims:
            if torch.rand(1).item() < self.p:
                x = torch.flip(x, dims=[dim])
        return x


class RandomIntensityJitter(nn.Module):
    """Subtle intensity scaling and additive Gaussian noise for normalized [0, 1] patches."""

    def __init__(self, scale_range: tuple[float, float] = (0.9, 1.1), noise_std: float = 0.02) -> None:
        super().__init__()
        self.scale_range = scale_range
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1, device=x.device).uniform_(*self.scale_range).item()
        x = x * scale
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return torch.clamp(x, 0.0, 1.0)


class ContrastiveDualViewTransform:
    """Produces two distinct stochastic augmentations of each 3D patch."""

    def __init__(self) -> None:
        self.flip = Random3DFlip(p=0.5)
        self.jitter = RandomIntensityJitter()

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v1 = self.jitter(self.flip(x.clone()))
        v2 = self.jitter(self.flip(x.clone()))
        return v1, v2
