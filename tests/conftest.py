"""
Pytest configuration and global environment setups.
"""

import os

# Prevent OpenBLAS memory allocation error on Windows multi-core systems
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
