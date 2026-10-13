"""
Smoke tests verifying repository scaffolding, interface contracts, and secret prevention.
"""

import re
from pathlib import Path

import numpy as np
import pytest
import torch

import copd_anomaly
from copd_anomaly.interfaces import (
    BaseEncoder,
    PatchRecord,
    RegistrationResult,
    ScanMetadata,
)


@pytest.mark.synthetic
def test_package_metadata():
    """Verify package import and version string."""
    assert hasattr(copd_anomaly, "__version__")
    assert copd_anomaly.__version__ == "0.1.0"


@pytest.mark.synthetic
def test_interface_dataclasses():
    """Verify that interface dataclasses instantiate with expected types and shapes."""
    metadata = ScanMetadata(
        patient_id="copd1",
        scan_id="copd1_insp",
        phase="insp",
        spacing_zyx=(2.5, 0.625, 0.625),
        shape_zyx=(121, 512, 512),
        hu_offset_verified=True,
        lung_median_hu=-886.0,
    )
    assert metadata.patient_id == "copd1"
    assert metadata.spacing_zyx == (2.5, 0.625, 0.625)

    # PatchRecord instantiation
    record = PatchRecord(
        patch_id="copd1_insp_001",
        patient_id="copd1",
        scan_id="copd1_insp",
        center_zyx=(60, 256, 256),
        lobe_id=1,
        lung_fraction=0.85,
        volume_weight=1.0,
        laa950_fraction=0.005,
    )
    assert record.lung_fraction >= 0.7
    assert record.lobe_id == 1

    # RegistrationResult instantiation with dummy shapes
    dummy_warped = np.zeros((10, 10, 10), dtype=np.float32)
    reg_result = RegistrationResult(
        warped_volume=dummy_warped,
        displacement_field=None,
        transform="dummy_transform",
        tre_stats={"mean_mm": 1.1, "sd_mm": 0.5},
    )
    assert reg_result.warped_volume.shape == (10, 10, 10)
    assert reg_result.tre_stats["mean_mm"] == 1.1


@pytest.mark.synthetic
def test_base_encoder_subclass():
    """Verify BaseEncoder protocol contract with synthetic tensors."""

    class SyntheticEncoder(BaseEncoder):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc_emb = torch.nn.Linear(1, 128)
            self.fc_proj = torch.nn.Linear(128, 64)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # x shape: (B, C, 50, 50, 50)
            b = x.shape[0]
            pooled = self.backbone(x).view(b, -1)
            z_emb = self.fc_emb(pooled)
            z_proj = self.fc_proj(z_emb)
            return z_emb, z_proj

    encoder = SyntheticEncoder()
    x = torch.randn(2, 1, 50, 50, 50)
    z_emb, z_proj = encoder(x)
    assert z_emb.shape == (2, 128)
    assert z_proj.shape == (2, 64)


@pytest.mark.synthetic
def test_no_secrets_in_tracked_files():
    """Ensure no Hugging Face tokens are hardcoded or tracked in repository files."""
    repo_root = Path(__file__).resolve().parent.parent
    hf_token_regex = re.compile(r"hf_[A-Za-z0-9]{20,}")

    target_dirs = ["src", "configs", "docs", "scripts", "notebooks", "tests"]
    for dir_name in target_dirs:
        dir_path = repo_root / dir_name
        if not dir_path.is_dir():
            continue
        for file_path in dir_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in {
                ".py",
                ".yaml",
                ".yml",
                ".md",
                ".toml",
                ".json",
                ".txt",
            }:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                matches = hf_token_regex.findall(content)
                assert len(matches) == 0, (
                    f"Potential Hugging Face secret detected in {file_path.relative_to(repo_root)}!"
                )

    # Also check root-level files
    for root_file in repo_root.glob("*.*"):
        if root_file.name in {".env.example", ".env.txt"}:
            continue
        if root_file.suffix in {".py", ".yaml", ".yml", ".md", ".toml", ".json", ".txt"}:
            content = root_file.read_text(encoding="utf-8", errors="ignore")
            matches = hf_token_regex.findall(content)
            assert len(matches) == 0, f"Potential Hugging Face secret detected in {root_file.name}!"
