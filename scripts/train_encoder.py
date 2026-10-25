"""
Train a 3D ResNet contrastive encoder using real patch shards.

Real training writes a best checkpoint, an epoch-by-epoch resumable checkpoint, and
CSV loss history. Synthetic five-step training is explicit and is never a research checkpoint.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from copd_anomaly.models.augmentations import ContrastiveDualViewTransform
from copd_anomaly.models.encoder import ResNet3DEncoder
from copd_anomaly.models.losses import NTXentLoss
from copd_anomaly.patches.dataset import ShardedPatchDataset
from copd_anomaly.utils.helpers import get_logger, set_seed

logger = get_logger("train_encoder")


def _atomic_torch_save(payload: dict[str, object], destination: Path) -> None:
    """Write, reload-verify, and atomically publish a checkpoint in its destination directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        check = torch.load(temporary, map_location="cpu", weights_only=False)
        if not isinstance(check, dict) or "training_mode" not in check:
            raise OSError(f"Checkpoint verification failed for {destination}.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


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
    resume_from: Path | None = None,
) -> Path:
    if epochs < 1 or batch_size < 2:
        raise ValueError("epochs must be >= 1 and batch_size must be >= 2 for NT-Xent.")
    if lr <= 0 or not torch.isfinite(torch.tensor(lr)):
        raise ValueError(f"lr must be finite and positive, got {lr}.")
    if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError(f"temperature must be finite and positive, got {temperature}.")
    path_parts = {part.lower() for part in checkpoint_dir.parts}
    if dry_run and "real" in path_parts:
        raise ValueError(
            "Synthetic dry-run checkpoints cannot be written under a real output directory."
        )
    if not dry_run and "synthetic" in path_parts:
        raise ValueError(
            "Real training checkpoints cannot be written under a synthetic output directory."
        )
    if not dry_run:
        if shards_dir is None:
            raise ValueError(
                "Real training requires --shards-dir; use --dry-run only for a smoke test."
            )
        manifest = shards_dir / "patch_manifest.parquet"
        if not manifest.is_file() and not (shards_dir / "patch_manifest.csv").is_file():
            raise FileNotFoundError(
                f"Patch manifest not found under {shards_dir}. Build real patches first, or pass "
                "--dry-run for a synthetic smoke test. No checkpoint was written."
            )
        if checkpoint_dir == Path("outputs/checkpoints"):
            checkpoint_dir = Path("outputs/real/checkpoints")
        if resume_from is not None and not resume_from.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
    elif checkpoint_dir == Path("outputs/checkpoints"):
        checkpoint_dir = Path("outputs/synthetic/checkpoints")

    set_seed(seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if device != "auto":
        device_obj = torch.device(device)
        if device_obj.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
    else:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training 3D contrastive encoder on device: %s", device_obj)

    dataset = None
    input_channels = 1
    if not dry_run:
        assert shards_dir is not None
        dataset = ShardedPatchDataset(shards_dir)
        if len(dataset) < batch_size:
            raise ValueError(
                f"Only {len(dataset)} patches are available; at least batch_size={batch_size} "
                "are required. No checkpoint was written."
            )
        sample_patch, _ = dataset[0]
        if sample_patch.ndim != 4:
            raise ValueError(
                f"Expected a patch shaped (C, Z, Y, X), got {tuple(sample_patch.shape)}."
            )
        if len(set(sample_patch.shape[1:])) != 1 or min(sample_patch.shape[1:]) < 16:
            raise ValueError(
                f"Expected a cubic patch with each spatial dimension >= 16, "
                f"got {tuple(sample_patch.shape)}."
            )
        input_channels = int(sample_patch.shape[0])
        if input_channels not in (1, 2):
            raise ValueError(
                f"Expected 1-channel inhale or 2-channel paired patch; got {input_channels}."
            )
        logger.info(
            "Patch dataset has %d samples and %d input channel(s).", len(dataset), input_channels
        )

    model = ResNet3DEncoder(
        in_channels=input_channels, base_filters=16, embedding_dim=128, projection_dim=64
    ).to(device_obj)
    criterion = NTXentLoss(temperature=temperature).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dual_transform = ContrastiveDualViewTransform()
    trained_steps = 0
    history: list[dict[str, float | int]] = []
    start_epoch = 0
    best_loss = float("inf")
    scheduler = None

    if dry_run:
        if resume_from is not None:
            raise ValueError("Resume is supported for real training only.")
        logger.warning(
            "SYNTHETIC DRY RUN ONLY: five random-input steps. This checkpoint is not a trained "
            "research model and must not be used for scientific results."
        )
        model.train()
        for step in range(5):
            dummy_batch = torch.rand(4, 1, 30, 30, 30, device=device_obj)
            v1, v2 = dual_transform(dummy_batch)
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            trained_steps += 1
            logger.info("Synthetic dry-run step %d/5 - loss %.4f", step + 1, loss.item())
    else:
        assert shards_dir is not None
        assert dataset is not None
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        if len(loader) == 0:
            raise ValueError(
                "No complete training batches are available; no checkpoint was written."
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        if resume_from is not None:
            try:
                checkpoint = torch.load(resume_from, map_location=device_obj, weights_only=False)
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot load resume checkpoint {resume_from}; it may be truncated or corrupt: {exc}"
                ) from exc
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Resume checkpoint must contain a dictionary: {resume_from}")
            if checkpoint.get("training_mode") != "real_patch_training":
                raise ValueError("Refusing to resume from a synthetic or unverified checkpoint.")
            required = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch"}
            missing = required - checkpoint.keys()
            if missing:
                raise ValueError(f"Resume checkpoint is incomplete; missing: {sorted(missing)}")
            if int(checkpoint.get("in_channels", 1)) != input_channels:
                raise ValueError(
                    f"Resume checkpoint expects {checkpoint.get('in_channels')} input channels, "
                    f"but current patches have {input_channels}."
                )
            expected_config = {
                "shards_dir": str(shards_dir),
                "batch_size": batch_size,
                "learning_rate": lr,
                "temperature": temperature,
                "seed": seed,
            }
            saved_config = checkpoint.get("config", {})
            mismatches = {
                key: (saved_config.get(key), value)
                for key, value in expected_config.items()
                if saved_config.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    "Resume settings differ from the saved checkpoint "
                    "(saved, requested): "
                    f"{mismatches}. Use the original settings or start a fresh run."
                )
            if int(checkpoint.get("epochs_requested", epochs)) != epochs:
                raise ValueError(
                    f"Checkpoint was configured for {checkpoint.get('epochs_requested')} epochs, "
                    f"but --epochs={epochs}. Resume with the original total epoch count."
                )
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_loss = float(checkpoint.get("best_loss", float("inf")))
            trained_steps = int(checkpoint.get("trained_steps", 0))
            history = list(checkpoint.get("history", []))
            logger.info("Resuming at epoch %d from %s", start_epoch + 1, resume_from)
            if start_epoch >= epochs:
                best_path = checkpoint_dir / "encoder_best.pt"
                if best_path.is_file():
                    logger.info("Requested epochs already completed; returning %s", best_path)
                    return best_path
                raise ValueError(
                    "Resume checkpoint is already at requested epochs but best checkpoint is missing."
                )

        for epoch in range(start_epoch, epochs):
            epoch_started = time.perf_counter()
            model.train()
            total_loss = 0.0
            n_batches = 0
            for batch_idx, (patches, metadata) in enumerate(loader, start=1):
                try:
                    patches = patches.to(device_obj, dtype=torch.float32)
                    if not torch.isfinite(patches).all():
                        raise ValueError("Input patch batch contains NaN or infinite values.")
                    v1, v2 = dual_transform(patches)
                    _, z1 = model(v1)
                    _, z2 = model(v2)
                    loss = criterion(z1, z2)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite contrastive loss: {loss.item()}.")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    for parameter in model.parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError("Non-finite gradient detected before optimizer step.")
                    optimizer.step()
                except Exception as exc:
                    scan_ids = metadata.get("scan_id", "unknown") if isinstance(metadata, dict) else "unknown"
                    logger.exception(
                        "Training failed at epoch %d/%d, batch %d/%d, scan_id(s)=%s",
                        epoch + 1,
                        epochs,
                        batch_idx,
                        len(loader),
                        scan_ids,
                    )
                    raise RuntimeError(
                        f"Training failed at epoch {epoch + 1}, batch {batch_idx}; "
                        f"scan_id(s)={scan_ids}. Latest completed epoch checkpoint is retained."
                    ) from exc
                total_loss += float(loss.item())
                n_batches += 1
                trained_steps += 1

            if n_batches == 0:
                raise RuntimeError("Training produced zero batches; no checkpoint was written.")
            avg_loss = total_loss / n_batches
            scheduler.step()
            if not torch.isfinite(torch.tensor(avg_loss)):
                raise FloatingPointError(
                    f"Epoch {epoch + 1} produced a non-finite mean loss: {avg_loss}."
                )
            epoch_seconds = time.perf_counter() - epoch_started
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_contrastive_loss": avg_loss,
                    "batches": n_batches,
                    "epoch_seconds": epoch_seconds,
                }
            )
            logger.info(
                "Epoch %d/%d - contrastive loss %.5f - %.1f seconds",
                epoch + 1,
                epochs,
                avg_loss,
                epoch_seconds,
            )

            payload = {
                "model_state_dict": model.state_dict(),
                "embedding_dim": model.embedding_dim,
                "projection_dim": model.projection_dim,
                "in_channels": model.in_channels,
                "training_mode": "real_patch_training",
                "trained_steps": trained_steps,
                "epochs_requested": epochs,
                "epoch": epoch,
                "best_loss": best_loss,
                "seed": seed,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history,
                "config": {
                    "shards_dir": str(shards_dir),
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "temperature": temperature,
                    "seed": seed,
                    "device": str(device_obj),
                },
            }
            if avg_loss < best_loss:
                best_loss = avg_loss
                payload["best_loss"] = best_loss
                _atomic_torch_save(payload, checkpoint_dir / "encoder_best.pt")
            _atomic_torch_save(payload, checkpoint_dir / "encoder_latest.pt")
            history_path = checkpoint_dir / "training_history.csv"
            history_tmp = history_path.with_name(f".{history_path.name}.tmp")
            try:
                pd.DataFrame(history).to_csv(history_tmp, index=False)
                # Read back the staged history to detect truncated or malformed writes.
                verified_history = pd.read_csv(history_tmp)
                if len(verified_history) != len(history):
                    raise OSError(
                        f"Training history verification failed: {len(verified_history)} rows "
                        f"for {len(history)} epochs."
                    )
                os.replace(history_tmp, history_path)
            finally:
                history_tmp.unlink(missing_ok=True)

    if dry_run:
        save_path = checkpoint_dir / "encoder_smoke_test.pt"
        _atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "embedding_dim": model.embedding_dim,
                "projection_dim": model.projection_dim,
                "in_channels": model.in_channels,
                "training_mode": "synthetic_dry_run",
                "trained_steps": trained_steps,
                "epochs_requested": 0,
                "seed": seed,
            },
            save_path,
        )
        logger.info("Saved synthetic smoke-test checkpoint to %s", save_path)
        return save_path

    save_path = checkpoint_dir / "encoder_best.pt"
    if not save_path.is_file():
        raise RuntimeError(f"Training ended without a best checkpoint: {save_path}")
    logger.info("Saved best real-training checkpoint to %s", save_path)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3D contrastive encoder.")
    parser.add_argument("--shards-dir", type=Path, default=Path("data/processed/patches"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or another torch device string"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-from", type=Path, default=None, help="Resume real training from encoder_latest.pt"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run five synthetic steps; not research training"
    )
    args = parser.parse_args()
    train_contrastive_encoder(
        shards_dir=args.shards_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        dry_run=args.dry_run,
        device=args.device,
        seed=args.seed,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
