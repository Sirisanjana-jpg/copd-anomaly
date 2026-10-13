"""
COPD-Anomaly: Self-supervised anomaly assessment of chest CT and lobe-wise exacerbation risk modeling.
"""

import os

# Prevent OpenBLAS memory allocation failures on Windows multi-core environments
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

__version__ = "0.1.0"
