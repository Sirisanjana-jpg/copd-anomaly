"""
Fit normal-lung density models (GMM or Normalizing Flow) on report-normal patches.

Enforces normal criteria: report negative for all 7 pulmonary conditions AND %LAA-950 < 0.01.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from copd_anomaly.anomaly.density import GaussianMixtureDensity, RealNVPFlowDensity
from copd_anomaly.models.encoder import ResNet3DEncoder
from copd_anomaly.patches.dataset import ShardedPatchDataset
from copd_anomaly.utils.helpers import get_logger, set_seed

logger = get_logger("fit_density")


def fit_density_estimator(
    shards_dir: Path | None = None,
    encoder_ckpt: Path | None = None,
    output_model_path: Path = Path("outputs/checkpoints/density_model.joblib"),
    model_type: str = "gmm",
    n_components: int = 8,
    dry_run: bool = False,
    seed: int = 42,
) -> Path:
    set_seed(seed)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run or shards_dir is None or not (shards_dir / "patch_manifest.parquet").exists():
        logger.info("Generating synthetic normal patch embeddings for dry-run fitting (N=100, D=64)...")
        # Synthetic Gaussian cluster representing normal lung tissue
        rng = np.random.RandomState(seed)
        z_normal = rng.randn(100, 64).astype(np.float32)

    else:
        logger.info(f"Loading normal patch embeddings from {shards_dir}...")
        dataset = ShardedPatchDataset(shards_dir, normal_only=True, normal_emphysema_cap=0.01)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = ResNet3DEncoder(in_channels=1, base_filters=16, embedding_dim=128, projection_dim=64).to(device)

        if encoder_ckpt and encoder_ckpt.exists():
            ckpt = torch.load(encoder_ckpt, map_location=device)
            encoder.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded pretrained encoder weights from {encoder_ckpt}")
        encoder.eval()

        embeddings = []
        with torch.no_grad():
            for patches, _ in loader:
                patches = patches.to(device)
                z_emb, _ = encoder(patches)
                embeddings.append(z_emb.cpu().numpy())

        z_normal = np.concatenate(embeddings, axis=0)
        logger.info(f"Extracted {len(z_normal)} normal embeddings (dim={z_normal.shape[1]}).")

    if model_type == "gmm":
        density_model = GaussianMixtureDensity(n_components=n_components, covariance_type="diag", random_state=seed)
        density_model.fit(z_normal)
        density_model.save(str(output_model_path))
    else:
        density_model = RealNVPFlowDensity(dim=z_normal.shape[1], n_layers=4, epochs=10)
        density_model.fit(z_normal)
        torch.save({"flow_state": density_model.flow.state_dict(), "dim": z_normal.shape[1]}, output_model_path)

    logger.info(f"Successfully saved fitted {model_type} density model to {output_model_path}")
    return output_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit normal-lung density estimator.")
    parser.add_argument("--shards-dir", type=Path, default=Path("data/processed/patches"))
    parser.add_argument("--encoder-ckpt", type=Path, default=Path("outputs/checkpoints/encoder_best.pt"))
    parser.add_argument("--output-model", type=Path, default=Path("outputs/checkpoints/density_model.joblib"))
    parser.add_argument("--model-type", type=str, default="gmm", choices=["gmm", "flow"])
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fit_density_estimator(
        shards_dir=args.shards_dir,
        encoder_ckpt=args.encoder_ckpt,
        output_model_path=args.output_model,
        model_type=args.model_type,
        n_components=args.components,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
