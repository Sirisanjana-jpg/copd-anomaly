"""
Normal-Lung Density Estimation and Normalizing Flows.

Implements:
1. GaussianMixtureDensity: GMM density estimator with diagonal/full covariance.
2. RealNVPFlowDensity: Deep invertible normalizing flow mapping embeddings to standard Gaussian.
3. Patch Anomaly Scorer: s(x) = -log p(z).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

from copd_anomaly.interfaces import AnomalyScorer, DensityModel
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("density_model")


class GaussianMixtureDensity(DensityModel):
    """Gaussian Mixture Model density estimator."""

    def __init__(
        self,
        n_components: int = 8,
        covariance_type: Literal["diag", "full", "tied", "spherical"] = "diag",
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.gmm: GaussianMixture | None = None
        self.training_mode: str | None = None
        self.fit_metadata: dict[str, object] = {}

    def fit(self, z_normal: np.ndarray | torch.Tensor) -> None:
        if isinstance(z_normal, torch.Tensor):
            z_normal = z_normal.detach().cpu().numpy()
        z_normal = np.asarray(z_normal)
        if z_normal.ndim != 2 or z_normal.shape[0] < 2 or z_normal.shape[1] < 1:
            raise ValueError(
                f"Expected at least two 2D embeddings for density fitting, got {z_normal.shape}."
            )
        if not np.isfinite(z_normal).all():
            raise ValueError("Density fitting embeddings contain NaN or infinite values.")

        n_samples, _ = z_normal.shape
        n_comp = min(self.n_components, max(1, n_samples // 5))

        logger.info(
            "Fitting GMM density model "
            f"(K={n_comp}, cov={self.covariance_type}) on {n_samples} normal patches..."
        )
        self.gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            max_iter=150,
        )
        self.gmm.fit(z_normal)
        self.fit_metadata["fit_sample_count"] = int(n_samples)
        if not self.gmm.converged_:
            raise RuntimeError(
                f"GMM did not converge after {self.gmm.n_iter_} iterations; "
                "do not use this density model. Review embedding quality and sample count."
            )
        logger.info("GMM fit converged successfully.")

    def log_prob(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.gmm is None:
            raise RuntimeError("GMM density model must be fitted before evaluating log_prob!")
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()
        return self.gmm.score_samples(z)

    def save(self, path: str) -> None:
        if self.gmm is None:
            raise RuntimeError("Cannot save a GMM density model before fitting.")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            joblib.dump(
                {
                    "gmm": self.gmm,
                    "params": {
                        "n_components": self.n_components,
                        "covariance_type": self.covariance_type,
                    },
                    "training_mode": self.training_mode,
                    "fit_metadata": self.fit_metadata,
                },
                temporary,
            )
            check = joblib.load(temporary)
            if check.get("gmm") is None or not hasattr(check["gmm"], "means_"):
                raise OSError(f"Density model verification failed for {destination}.")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str) -> GaussianMixtureDensity:
        data = joblib.load(path)
        obj = cls(**data["params"])
        obj.gmm = data["gmm"]
        obj.training_mode = data.get("training_mode")
        obj.fit_metadata = data.get("fit_metadata", {})
        return obj


class AffineCouplingLayer(nn.Module):
    """Coupling layer for RealNVP normalizing flow."""

    def __init__(self, dim: int, mask: torch.Tensor, hidden_dim: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, dim * 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_masked = x * self.mask
        out = self.net(x_masked)
        s, t = out.chunk(2, dim=-1)
        s = torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)

        y = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        y_masked = y * self.mask
        out = self.net(y_masked)
        s, t = out.chunk(2, dim=-1)
        s = torch.tanh(s) * (1.0 - self.mask)
        t = t * (1.0 - self.mask)

        x = y_masked + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        return x


class RealNVPFlow(nn.Module):
    """RealNVP normalizing flow architecture."""

    def __init__(self, dim: int, n_layers: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()

        for i in range(n_layers):
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[: dim // 2] = 1.0
            else:
                mask[dim // 2 :] = 1.0
            self.layers.append(AffineCouplingLayer(dim, mask, hidden_dim=hidden_dim))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_det_sum = torch.zeros(x.size(0), device=x.device)
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_sum += log_det

        log_prior = -0.5 * (np.log(2.0 * np.pi) * self.dim + (z**2).sum(dim=-1))
        return log_prior + log_det_sum


class RealNVPFlowDensity(DensityModel):
    """Normalizing Flow density estimator wrapping RealNVPFlow."""

    def __init__(
        self,
        dim: int = 128,
        n_layers: int = 4,
        epochs: int = 50,
        lr: float = 1e-3,
    ) -> None:
        self.dim = dim
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.flow: RealNVPFlow | None = None

    def fit(self, z_normal: np.ndarray | torch.Tensor) -> None:
        if isinstance(z_normal, np.ndarray):
            z_tensor = torch.from_numpy(z_normal).float()
        else:
            z_tensor = z_normal.float().detach().cpu()

        self.dim = z_tensor.shape[1]
        self.flow = RealNVPFlow(dim=self.dim, n_layers=self.n_layers)

        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.lr)
        self.flow.train()

        n_samples = z_tensor.size(0)
        batch_size = min(64, n_samples)
        logger.info(f"Fitting RealNVP flow on {n_samples} embeddings (dim={self.dim})...")

        for _ in range(self.epochs):
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                batch = z_tensor[perm[i : i + batch_size]]
                loss = -self.flow.log_prob(batch).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def log_prob(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.flow is None:
            raise RuntimeError("Flow density model must be fitted first!")
        if isinstance(z, np.ndarray):
            z_tensor = torch.from_numpy(z).float()
        else:
            z_tensor = z.float().detach().cpu()

        self.flow.eval()
        with torch.no_grad():
            lp = self.flow.log_prob(z_tensor)
        return lp.cpu().numpy()


class PatchAnomalyScorer(AnomalyScorer):
    """Computes patch anomaly scores s(x) = -log p(f(x))."""

    def __init__(self, density_model: DensityModel) -> None:
        self.density_model = density_model

    def score_patches(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        """Computes negative log-likelihood anomaly score. Higher score = more anomalous."""
        log_p = self.density_model.log_prob(z)
        scores = -log_p
        return scores.astype(np.float32)
