"""
Train 3D ResNet Contrastive Encoder via NT-Xent Self-Supervised Learning.

Optimizes 3D patch representations to learn invariant pulmonary tissue features,
saving the resulting feature extractor for downstream normal-lung density modeling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from copd_anomaly.models.augmentations import ContrastiveDualViewTransform
from copd_anomaly.models.encoder import ResNet3DEncoder
from copd_anomaly.models.losses import NTXentLoss
from copd_anomaly.patches.dataset import ShardedPatchDataset
from copd_anomaly.utils.helpers import get_logger, set_seed

logger = get_logger("train_encoder")


def train_contrastive_encoder(
    shards_dir: Path | None = None,
    checkpoint_dir: Path = Path("outputs/checkpoints"),
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-4,
    temperature: float = 0.07,
    dry_run: bool = False,
    device: str = "auto",
    seed: int = 42,
) -> Path:
    set_seed(seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else "cpu")
    logger.info(f"Training 3D Contrastive Encoder on device: {device_obj}")

    model = ResNet3DEncoder(in_channels=1, base_filters=16, embedding_dim=128, projection_dim=64).to(
        device_obj
    )
    criterion = NTXentLoss(temperature=temperature).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    dual_transform = ContrastiveDualViewTransform()

    if dry_run or shards_dir is None or not (shards_dir / "patch_manifest.parquet").exists():
        logger.info("Executing dry-run / synthetic pretraining loop (5 steps)...")
        model.train()
        for step in range(5):
            # Synthetic 3D patch batch (B=4, C=1, 30, 30, 30) for fast CPU validation
            dummy_batch = torch.rand(4, 1, 30, 30, 30, device=device_obj)
            v1, v2 = dual_transform(dummy_batch)
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info(f"  Dry-run Step {step + 1}/5 - Loss: {loss.item():.4f}")

    else:
        dataset = ShardedPatchDataset(shards_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_loss = float("inf")
        model.train()

        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            for patches, _ in loader:
                patches = patches.to(device_obj)
                v1, v2 = dual_transform(patches)
                _, z1 = model(v1)
                _, z2 = model(v2)
                loss = criterion(z1, z2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            scheduler.step()
            logger.info(f"Epoch {epoch + 1}/{epochs} - Contrastive Loss: {avg_loss:.4f}")

    save_path = checkpoint_dir / "encoder_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": model.embedding_dim,
            "projection_dim": model.projection_dim,
            "in_channels": model.in_channels,
        },
        save_path,
    )
    logger.info(f"Saved encoder checkpoint to {save_path}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3D contrastive encoder.")
    parser.add_argument("--shards-dir", type=Path, default=Path("data/processed/patches"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_contrastive_encoder(
        shards_dir=args.shards_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
