"""Synthetic COPD cohort generation, multimodal fusion, and uncertainty estimation."""

from copd_anomaly.synthetic.cohort_generator import (
    SYNTHETIC_BANNER,
    generate_synthetic_copd_cohort,
)
from copd_anomaly.synthetic.multimodal import (
    ExacerbationRiskHead,
    MultimodalFusionModel,
    MultimodalRiskPipeline,
)

__all__ = [
    "generate_synthetic_copd_cohort",
    "SYNTHETIC_BANNER",
    "MultimodalFusionModel",
    "ExacerbationRiskHead",
    "MultimodalRiskPipeline",
]
