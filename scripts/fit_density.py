"""
Fit a normal-lung density model from embeddings of eligible real patch shards.

Synthetic fitting is permitted only with explicit --dry-run and is written to the
synthetic output area by default.
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
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")
    if model_type not in {"gmm", "flow"}:
        raise ValueError("model_type must be 'gmm' or 'flow'.")

    if dry_run:
        if output_model_path == Path("outputs/checkpoints/density_model.joblib"):
            output_model_path = Path("outputs/synthetic/checkpoints/density_model.joblib")
    else:
        if shards_dir is None:
            raise ValueError("Real density fitting requires --shards-dir; use --dry-run for a synthetic smoke test.")
        manifest = shards_dir / "patch_manifest.parquet"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"Patch manifest not found: {manifest}. Build real patches before fitting density. "
                "No model was written."
            )
        if encoder_ckpt is None or not encoder_ckpt.is_file():
            raise FileNotFoundError(
                f"A trained encoder checkpoint is required for real density fitting: {encoder_ckpt}. "
                "Run real encoder training first; no model was written."
            )
        if "synthetic" in encoder_ckpt.parts:
            raise ValueError("A synthetic smoke-test checkpoint cannot be used for real density fitting.")
        if output_model_path == Path("outputs/checkpoints/density_model.joblib"):
            output_model_path = Path("outputs/real/checkpoints/density_model.joblib")

    set_seed(seed)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.warning("SYNTHETIC DRY RUN ONLY: fitting density on random embeddings, not CT data.")
        rng = np.random.RandomState(seed)
        z_normal = rng.randn(100, 64).astype(np.float32)
    else:
        assert shards_dir is not None and encoder_ckpt is not None
        dataset = ShardedPatchDataset(shards_dir, normal_only=True, normal_emphysema_cap=0.01)
        if len(dataset) == 0:
            raise ValueError("No eligible normal patches found. Check report labels and LAA-950 metadata.")
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = ResNet3DEncoder(in_channels=1, base_filters=16, embedding_dim=128, projection_dim=64).to(device)
        checkpoint = torch.load(encoder_ckpt, map_location=device, weights_only=False)
        if checkpoint.get("training_mode") == "synthetic_dry_run":
            raise ValueError("Refusing synthetic encoder checkpoint for real density fitting.")
        if "model_state_dict" not in checkpoint:
            raise ValueError(f"Invalid encoder checkpoint (missing model_state_dict): {encoder_ckpt}")
        encoder.load_state_dict(checkpoint["model_state_dict"])
        encoder.eval()
        embeddings = []
        with torch.no_grad():
            for patches, _ in loader:
                _, z_emb = encoder(patches.to(device, dtype=torch.float32))
                embeddings.append(z_emb.cpu().numpy())
        if not embeddings:
            raise ValueError("No embeddings extracted from eligible patches.")
        z_normal = np.concatenate(embeddings, axis=0)
        if z_normal.ndim != 2 or z_normal.shape[0] < max(2, n_components):
            raise ValueError(
                f"Need at least max(2, n_components)={max(2, n_components)} embeddings; got {z_normal.shape}."
            )
        logger.info("Extracted %d real embeddings (dim=%d).", len(z_normal), z_normal.shape[1])

    if model_type == "gmm":
        density_model = GaussianMixtureDensity(n_components=n_components, covariance_type="diag", random_state=seed)
        density_model.fit(z_normal)
        density_model.save(str(output_model_path))
    else:
        density_model = RealNVPFlowDensity(dim=z_normal.shape[1], n_layers=4, epochs=10)
        density_model.fit(z_normal)
        torch.save(
            {"flow_state": density_model.flow.state_dict(), "dim": z_normal.shape[1],
             "training_mode": "synthetic_dry_run" if dry_run else "real_patch_training"},
            output_model_path,
        )
    logger.info("Saved %s density model to %s", "synthetic" if dry_run else "real", output_model_path)
    return output_model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit normal-lung density estimator.")
    parser.add_argument("--shards-dir", type=Path, default=Path("data/processed/patches"))
    parser.add_argument("--encoder-ckpt", type=Path, default=Path("outputs/real/checkpoints/encoder_best.pt"))
    parser.add_argument("--output-model", type=Path, default=Path("outputs/checkpoints/density_model.joblib"))
    parser.add_argument("--model-type", type=str, default="gmm", choices=["gmm", "flow"])
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Fit on random embeddings; not research output")
    args = parser.parse_args()
    fit_density_estimator(
        shards_dir=args.shards_dir,
        encoder_ckpt=args.encoder_ckpt,
        output_model_path=args.output_model,
        model_type=args.model_type,
        n_components=args.components,
        dry_run=args.dry_run,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
