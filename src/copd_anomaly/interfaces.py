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
    phase: str  # 'insp' or 'exp'
    spacing_zyx: tuple[float, float, float]  # (sz, sy, sx) in mm
    shape_zyx: tuple[int, int, int]  # (Z, Y, X)
    origin_zyx: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    hu_offset_verified: bool = False
    lung_median_hu: float | None = None


@dataclass
class RegistrationResult:
    """Output of non-rigid registration between expiration and inspiration scans."""

    warped_volume: np.ndarray  # Shape: (Z, Y, X), dtype: float32, aligned with fixed
    displacement_field: np.ndarray | None  # Shape: (Z, Y, X, 3), dtype: float32 (dx, dy, dz)
    transform: Any  # Transform object (e.g. SimpleITK.Transform)
    tre_stats: dict[str, float] | None = None  # Optional TRE evaluation metrics


@dataclass
class PatchRecord:
    """Metadata accompanying an extracted 3D patch."""

    patch_id: str
    patient_id: str
    scan_id: str
    center_zyx: tuple[int, int, int]
    lobe_id: int | None  # 1: RUL, 2: RML, 3: RLL, 4: LUL, 5: LLL, 0/None: whole lung / unassigned
    lung_fraction: float  # Fraction of voxels inside lung mask (>= 0.7)
    volume_weight: float  # Proportional sampling weight
    laa950_fraction: float  # Fraction of patch voxels < -950 HU


@runtime_checkable
class DataSource(Protocol):
    """Protocol for loading raw CT volumes and landmark annotations."""

    def load_scan(self, case_id: str, phase: str) -> tuple[np.ndarray, ScanMetadata]:
        """
        Loads raw volume data and spatial geometry.

        Args:
            case_id: Identifier of the case (e.g. 'copd1').
            phase: 'insp' for inspiratory or 'exp' for expiratory.

        Returns:
            volume: np.ndarray, shape (Z, Y, X), dtype float32, in Hounsfield Units (HU).
            metadata: ScanMetadata object.
        """
        ...

    def load_landmarks(self, case_id: str, phase: str) -> np.ndarray:
        """
        Loads anatomical landmark coordinates.

        Args:
            case_id: Identifier of the case.
            phase: 'insp' or 'exp'.

        Returns:
            landmarks: np.ndarray, shape (N_landmarks, 3), dtype float32.
                       Coordinates as (x, y, z) in voxel indices or physical mm.
        """
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
        """
        Resamples a 3D volume to isotropic target spacing.

        Args:
            volume: np.ndarray, shape (Z, Y, X), dtype float32.
            current_spacing: (sz, sy, sx) in mm.
            target_spacing: Target spacing (sz, sy, sx) in mm (default: (1.0, 1.0, 1.0)).

        Returns:
            resampled_volume: np.ndarray, shape (Z', Y', X'), dtype float32.
            new_spacing: (1.0, 1.0, 1.0).
        """
        ...

    def window_and_normalize(
        self,
        volume: np.ndarray,
        window_hu: tuple[float, float] = (-1024.0, 0.0),
    ) -> np.ndarray:
        """
        Clips HU intensities and scales to [0, 1] float32 or [0, 255] uint8.

        Args:
            volume: np.ndarray, shape (Z, Y, X), dtype float32 in HU.
            window_hu: (min_hu, max_hu) clipping window.

        Returns:
            normalized: np.ndarray, shape (Z, Y, X), dtype float32 in [0, 1].
        """
        ...


@runtime_checkable
class Segmenter(Protocol):
    """Protocol for lung parenchyma and lobe segmentation."""

    def segment(
        self,
        volume: np.ndarray,
        spacing_zyx: tuple[float, float, float],
    ) -> np.ndarray:
        """
        Generates binary lung mask or multi-label 5-lobe mask.

        Args:
            volume: np.ndarray, shape (Z, Y, X), dtype float32 in HU.
            spacing_zyx: (sz, sy, sx) in mm.

        Returns:
            mask: np.ndarray, shape (Z, Y, X), dtype uint8.
                  0 = background,
                  1 = Right Upper Lobe (RUL),
                  2 = Right Middle Lobe (RML),
                  3 = Right Lower Lobe (RLL),
                  4 = Left Upper Lobe (LUL),
                  5 = Left Lower Lobe (LLL).
                  (If lobe segmenter unavailable, 1 = whole lung).
        """
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
        """
        Registers moving (exhale) volume to fixed (inhale) volume.

        Args:
            fixed_volume: (Z, Y, X) float32 in HU (inhale).
            moving_volume: (Z_m, Y_m, X_m) float32 in HU (exhale).
            fixed_mask: (Z, Y, X) uint8 lung mask.
            moving_mask: (Z_m, Y_m, X_m) uint8 lung mask.
            spacing_zyx: (sz, sy, sx) in mm.

        Returns:
            RegistrationResult containing warped volume of shape (Z, Y, X) and transform.
        """
        ...

    def transform_points(self, points: np.ndarray, transform: Any) -> np.ndarray:
        """
        Transforms moving point coordinates to fixed space.

        Args:
            points: (N, 3) float32 coordinates (x, y, z).
            transform: Fitted transform object.

        Returns:
            warped_points: (N, 3) float32 transformed coordinates.
        """
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
        """
        Extracts 3D volumetric patches restricted to lung mask.

        Args:
            volume_insp: (Z, Y, X) float32 or uint8.
            volume_exp: Optional (Z, Y, X) registered exhale.
            lung_mask: (Z, Y, X) uint8.
            lobe_mask: Optional (Z, Y, X) uint8 (1..5).
            patch_size: Voxel edge length (default: 50).
            overlap: Patch overlap fraction (0.0 or 0.2).
            min_lung_fraction: Minimum fraction of lung voxels in patch (>= 0.7).
            max_patches_per_scan: Maximum random patches retained per subject (e.g. 100).

        Returns:
            patches: np.ndarray, shape (N_patches, C, 50, 50, 50),
                     where C = 1 (insp only) or C = 2 (insp + registered exp).
            records: list of N_patches PatchRecord objects.
        """
        ...


class BaseEncoder(nn.Module):
    """Base neural network encoder for 3D patches."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: torch.Tensor, shape (B, C, 50, 50, 50), float32, C in {1, 2}.

        Returns:
            z_emb: torch.Tensor, shape (B, D), latent representation for anomaly scoring.
            z_proj: torch.Tensor, shape (B, d_proj), normalized projection for NT-Xent loss.
        """
        raise NotImplementedError


@runtime_checkable
class DensityModel(Protocol):
    """Protocol for fitting normal-lung distribution p(z) and computing log probabilities."""

    def fit(self, z_normal: np.ndarray | torch.Tensor) -> None:
        """
        Fits distribution on embeddings of normal control patches.

        Args:
            z_normal: np.ndarray or torch.Tensor, shape (N_normal, D), float32.
                      From report-normal scans with patch %LAA-950 < 0.01.
        """
        ...

    def log_prob(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Evaluates log probability density log p(z).

        Args:
            z: np.ndarray or torch.Tensor, shape (N_patches, D), float32.

        Returns:
            log_p: np.ndarray, shape (N_patches,), float32.
        """
        ...


@runtime_checkable
class AnomalyScorer(Protocol):
    """Protocol for computing patch anomaly scores s(x) = -log p(f(x))."""

    def score_patches(self, z: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Computes non-negative patch anomaly scores.

        Args:
            z: shape (N_patches, D), float32.

        Returns:
            scores: shape (N_patches,), float32.
        """
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
        """
        Aggregates patch anomaly scores into a scalar patient score.

        Args:
            patch_scores: shape (N_patches,), float32.
            volume_weights: Optional shape (N_patches,), float32.
            lobe_ids: Optional shape (N_patches,), uint8.

        Returns:
            S_X: scalar float anomaly score.
        """
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
        """
        Constructs comprehensive feature vector.

        Returns:
            features: Dictionary containing:
                global_anomaly_mean, global_anomaly_p90,
                rul_anomaly_mean, rml_anomaly_mean, rll_anomaly_mean,
                lul_anomaly_mean, lll_anomaly_mean,
                upper_lower_ratio, left_right_asymmetry,
                heterogeneity_index (std of 5 lobe means),
                and lobe-level PRM / %LAA metrics.
        """
        ...


class FusionModel(nn.Module):
    """Combines imaging and clinical features."""

    def forward(self, x_ct: torch.Tensor, x_clin: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_ct: torch.Tensor, shape (B, D_ct), float32.
            x_clin: torch.Tensor, shape (B, D_clin), float32.

        Returns:
            h: torch.Tensor, shape (B, D_fused), float32.
        """
        raise NotImplementedError


class RiskHead(nn.Module):
    """Predicts 12-month exacerbation probability or hazard."""

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: torch.Tensor, shape (B, D_fused), float32.

        Returns:
            risk: torch.Tensor, shape (B, 1), float32 logit or log hazard.
        """
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
