"""
Multimodal Fusion Network, Exacerbation Risk Head, and Monte Carlo Uncertainty Estimation.

Combines high-dimensional 3D CT representations with clinical covariates and spirometry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from copd_anomaly.interfaces import FusionModel, RiskHead


class MultimodalFusionModel(FusionModel):
    """
    Gated Cross-Modal Fusion combining CT imaging embeddings and clinical tabular features.
    """

    def __init__(
        self,
        d_ct: int = 128,
        d_clin: int = 8,
        d_hidden: int = 64,
        d_fused: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.d_ct = d_ct
        self.d_clin = d_clin
        self.d_fused = d_fused

        # Projection for imaging features
        self.ct_proj = nn.Sequential(
            nn.Linear(d_ct, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Projection for clinical tabular features
        self.clin_proj = nn.Sequential(
            nn.Linear(d_clin, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Gated Multi-Modal Fusion
        self.gate = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(d_hidden, d_fused),
            nn.BatchNorm1d(d_fused),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_ct: torch.Tensor, x_clin: torch.Tensor) -> torch.Tensor:
        """
        Combines CT and clinical features into a unified latent vector.

        Args:
            x_ct: (B, D_ct) float32.
            x_clin: (B, D_clin) float32.

        Returns:
            h: (B, D_fused) float32.
        """
        h_ct = self.ct_proj(x_ct)
        h_clin = self.clin_proj(x_clin)

        # Compute dynamic gating weights between imaging and clinical streams
        concat = torch.cat([h_ct, h_clin], dim=-1)
        g = self.gate(concat)

        fused = g * h_ct + (1.0 - g) * h_clin
        return self.out_proj(fused)


class ExacerbationRiskHead(RiskHead):
    """
    Predicts 12-month exacerbation probability with Monte Carlo Dropout for epistemic uncertainty.
    """

    def __init__(self, d_fused: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_fused, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Returns exacerbation logit.

        Args:
            h: (B, D_fused) float32.

        Returns:
            logit: (B, 1) float32.
        """
        return self.classifier(h)


class MultimodalRiskPipeline(nn.Module):
    """End-to-end multimodal risk predictor with MC-Dropout uncertainty estimation."""

    def __init__(
        self,
        d_ct: int = 128,
        d_clin: int = 8,
        d_fused: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.fusion = MultimodalFusionModel(
            d_ct=d_ct, d_clin=d_clin, d_hidden=64, d_fused=d_fused, dropout=dropout
        )
        self.head = ExacerbationRiskHead(d_fused=d_fused, dropout=dropout)

    def forward(self, x_ct: torch.Tensor, x_clin: torch.Tensor) -> torch.Tensor:
        h = self.fusion(x_ct, x_clin)
        return self.head(h)

    def predict_with_uncertainty(
        self,
        x_ct: torch.Tensor,
        x_clin: torch.Tensor,
        n_mc_samples: int = 30,
    ) -> dict[str, torch.Tensor]:
        """
        Computes mean predicted probability and epistemic uncertainty via Monte Carlo Dropout.
        """
        self.train()  # Keep dropout active during inference

        mc_probs = []
        with torch.no_grad():
            for _ in range(n_mc_samples):
                logits = self.forward(x_ct, x_clin)
                probs = torch.sigmoid(logits)
                mc_probs.append(probs)

        # Shape: (M, B, 1)
        stacked = torch.stack(mc_probs, dim=0)

        mean_prob = stacked.mean(dim=0).squeeze(-1)  # (B,)
        epistemic_var = stacked.var(dim=0).squeeze(-1)  # (B,)
        q_low = torch.quantile(stacked, 0.05, dim=0).squeeze(-1)
        q_high = torch.quantile(stacked, 0.95, dim=0).squeeze(-1)

        return {
            "mean_risk": mean_prob,
            "epistemic_uncertainty": epistemic_var,
            "ci90_low": q_low,
            "ci90_high": q_high,
        }
