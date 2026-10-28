"""
3D Convolutional Residual Encoder and Contrastive Projection Head.

Implements a 3D ResNet backbone tailored for volumetric CT patches (50x50x50),
supporting single-channel (inspiration) or dual-channel (inspiration + co-registered expiration)
inputs. Outputs latent representations for density estimation and normalized projection vectors
for NT-Xent contrastive learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from copd_anomaly.interfaces import BaseEncoder


class BasicBlock3D(nn.Module):
    """3D Residual Block."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet3DEncoder(BaseEncoder):
    """
    3D ResNet patch encoder with contrastive projection head.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 32,
        embedding_dim: int = 128,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim

        # Initial 3D Conv
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(base_filters),
            nn.ReLU(inplace=True),
        )

        # 3 Stages of residual blocks downsampling spatial resolution: 50 -> 25 -> 13 -> 7
        self.layer1 = self._make_layer(base_filters, base_filters, num_blocks=2, stride=2)
        self.layer2 = self._make_layer(base_filters, base_filters * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(base_filters * 2, base_filters * 4, num_blocks=2, stride=2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Latent embedding linear projection
        conv_out_features = base_filters * 4
        self.fc_emb = nn.Sequential(
            nn.Linear(conv_out_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Contrastive Projection Head: 2-layer MLP
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
        )

    def _make_layer(
        self, in_planes: int, out_planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock3D(in_planes, out_planes, stride=s))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, C, 50, 50, 50) float32 tensor.

        Returns:
            z_emb: (B, embedding_dim) representation for anomaly density modeling.
            z_proj: (B, projection_dim) unit-norm projection for contrastive loss.
        """
        out = self.init_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        pooled = self.global_pool(out).flatten(1)  # (B, conv_out_features)
        z_emb = self.fc_emb(pooled)  # (B, embedding_dim)

        z_proj_unnorm = self.projection_head(z_emb)  # (B, projection_dim)
        z_proj = F.normalize(z_proj_unnorm, p=2, dim=-1)

        return z_emb, z_proj
