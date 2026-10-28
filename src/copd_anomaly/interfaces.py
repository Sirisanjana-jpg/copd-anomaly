"""
Formal interfaces and protocols for COPD Anomaly Assessment and Exacerbation Risk Modeling.

Defines the contract and exact tensor shapes for all modular components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn


@dataclass
class ScanMetadata:
    """Metadata describing CT scan geometry and provenance."""

    patient_id: str
    scan_id: str
    phase: str
    spacing_zyx: tuple[float, float, float]
    shape_zyx: tuple[int, int, int]
    origin_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    hu_offset_verified: bool = False
    lung_median_hu: float | None = None


@dataclass
class RegistrationResult:
    """Output of non-rigid registration between expiration and inspiration scans."""

    warped_volume: np.ndarray
    displacement_field: np.ndarray | None
    transform: Any
    tre_stats: dict[str, float] | None = None


@dataclass
class PatchRecord:
    """Metadata accompanying an extracted 3D patch."""

    patch_id: str
    patient_id: str
    scan_id: str
    center_zyx: tuple[int, int, int]
    lobe_id: int | None
    lung_fraction: float
    volume_weight: float
    laa950_fraction: float


@runtime_checkable
class DataSource(Protocol):
    """Protocol for loading raw CT volumes and landmark annotations."""

    def load_scan(self, case_id: str, phase: str) -> tuple[np.ndarray, ScanMetadata]:
        """Loads raw volume data and spatial geometry."""
        ...

    def load_landmarks(self, case_id: str, phase: str) -> np.ndarray:
        """Loads anatomical landmark coordinates."""
        ...


@runtime_checkable
class Preprocessor(Protocol):
    """Protocol for intensity normalization and spatial resampling."""

    def resample(
        self,
        volume: np.ndarray,
        current_spacing: tuple[float, float, float],
        target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> tuple[np.ndarray, tuple[float, float, float]]:
        """Resamples a 3D volume to isotropic target spacing."""
        ...

    def window_and_normalize(
        self,
        volume: np.ndarray,
        window_hu: tuple[float, float] = (-1024.0, 0.0),
    ) -> np.ndarray:
        """Clips HU intensities and scales to [0, 1] float32 or [0, 255] uint8."""
        ...


@runtime_checkable
class Segmenter(Protocol):
    """Protocol for lung parenchyma and lobe segmentation."""

    def segment(
        self,
        volume: np.ndarray,
        spacing_zyx: tuple[float, float, float],
    ) -> np.ndarray:
        """Generates binary lung mask or multi-label 5-lobe mask."""
        ...


@runtime_checkable
class Registrar(Protocol):
    """Protocol for non-rigid registration between expiratory and inspiratory CT scans."""

    def register(
        self,
        fixed_volume: np.ndarray,
        moving_volume: np.ndarray,
        fixed_mask: np.ndarray,
        moving_mask: np.ndarray,
        spacing_zyx: tuple[float, float, float],
    ) -> RegistrationResult:
        """Registers moving volume to fixed volume."""
        ...

    def transform_points(self, points: np.ndarray, transform: Any) -> np.ndarray:
        """Transforms moving point coordinates to fixed space."""
        ...


@runtime_checkable
class PatchSampler(Protocol):
    """Protocol for extracting 3D patches from lung parenchyma."""

    def extract_patches(
        self,
        volume_insp: np.ndarray,
        volume_exp: np.ndarray | None,
        lung_mask: np.ndarray,
        lobe_mask: np.ndarray | None,
        patch_size: int = 50,
        overlap: float = 0.2,
        min_lung_fraction: float = 0.7,
        max_patches_per_scan: int = 100,
    ) -> tuple[np.ndarray, list[PatchRecord]]:
        """Extracts 3D volumetric patches restricted to lung mask."""
        ...


class BaseEncoder(nn.Module):
    """Base neural network encoder for 3D patches."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        raise NotImplementedError


@runtime_checkable
class DensityModel(Protocol):
    """Protocol for fitting normal-lung distribution p(z) and computing log probabilities."""

    def fit(self, z_normal: np.ndarray | torch.Tensor) -> None:
        """Fits distribution on embeddings of normal control patches."""
        ...

    def log_prob(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        """Evaluates log probability density log p(z)."""
        ...


@runtime_checkable
class AnomalyScorer(Protocol):
    """Protocol for computing patch anomaly scores s(x) = -log p(f(x))."""

    def score_patches(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        """Computes non-negative patch anomaly scores."""
        ...


@runtime_checkable
class Aggregator(Protocol):
    """Protocol for aggregating patch scores to scan-level score S(X)."""

    def aggregate(
        self,
        patch_scores: np.ndarray,
        volume_weights: np.ndarray | None = None,
        lobe_ids: np.ndarray | None = None,
    ) -> float:
        """Aggregates patch anomaly scores into a scalar patient score."""
        ...


@runtime_checkable
class FeatureBuilder(Protocol):
    """Protocol for extracting global and lobe-wise quantitative features."""

    def build_features(
        self,
        patch_scores: np.ndarray,
        records: list[PatchRecord],
        prm_stats: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Constructs comprehensive feature vector."""
        ...


class FusionModel(nn.Module):
    """Combines imaging and clinical features."""

    def forward(self, x_ct: torch.Tensor, x_clin: torch.Tensor) -> torch.Tensor:
        """Forward pass for fusion model."""
        raise NotImplementedError


class RiskHead(nn.Module):
    """Predicts 12-month exacerbation probability or hazard."""

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass for risk head."""
        raise NotImplementedError


@runtime_checkable
class Evaluator(Protocol):
    """Protocol for computing registration, classification, and survival metrics."""

    def evaluate_tre(
        self,
        fixed_landmarks: np.ndarray,
        warped_landmarks: np.ndarray,
        spacing_zyx: tuple[float, float, float],
    ) -> dict[str, float]:
        """Computes mean and std Target Registration Error (TRE) in mm."""
        ...

    def evaluate_proxy_auc(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        n_bootstraps: int = 1000,
    ) -> dict[str, Any]:
        """Computes AUROC, AUPRC with bootstrap 95% confidence intervals."""
        ...
