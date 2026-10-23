"""
NT-Xent (Normalized Temperature-scaled Cross Entropy) Contrastive Loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Computes NT-Xent loss between two augmented views of the same batch.

        Args:
            z_i: (B, D) L2-normalized projection vectors of view 1.
            z_j: (B, D) L2-normalized projection vectors of view 2.

        Returns:
            Scalar contrastive loss.
        """
        batch_size = z_i.size(0)
        device = z_i.device

        # Concatenate views: (2B, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Compute cosine similarity matrix: (2B, 2B)
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Create labels: for i in [0..B-1], positive is i + B; for i in [B..2B-1], positive is i - B
        labels = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=device),
                torch.arange(0, batch_size, device=device),
            ],
            dim=0,
        )

        # Mask out self-similarities (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss
