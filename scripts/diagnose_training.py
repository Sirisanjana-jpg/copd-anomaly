"""Preflight diagnostics for real CT patch shards before expensive GPU training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from copd_anomaly.patches.dataset import ShardedPatchDataset
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("diagnose_training")


def _existing_parent(path: Path) -> Path:
    """Return the closest existing parent for disk-usage checks on new output paths."""
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"No existing parent directory for {path}.")
        candidate = candidate.parent
    return candidate


def diagnose_training_inputs(
    shards_dir: Path,
    checkpoint_dir: Path,
    min_free_gb: float = 5.0,
    require_cuda: bool = False,
) -> dict[str, Any]:
    """Validate inputs and report environment details without starting training."""
    if min_free_gb < 0:
        raise ValueError(f"min_free_gb must be >= 0, got {min_free_gb}.")
    if not shards_dir.is_dir():
        raise FileNotFoundError(f"Patch shard directory does not exist: {shards_dir}")

    report: dict[str, Any] = {
        "status": "failed",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "shards_dir": str(shards_dir.resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "checks": [],
        "errors": [],
        "warnings": [],
    }

    for package in ("numpy", "pandas", "pyarrow", "nibabel", "SimpleITK"):
        try:
            report.setdefault("package_versions", {})[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["errors"].append(f"Required package is missing: {package}")

    if require_cuda and not torch.cuda.is_available():
        report["errors"].append("--require-cuda was set but PyTorch cannot access CUDA.")
    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory

    for path in (shards_dir, checkpoint_dir):
        usage = shutil.disk_usage(_existing_parent(path))
        free_gb = usage.free / (1024**3)
        report.setdefault("disk", {})[str(path)] = {
            "free_gb": round(free_gb, 2),
            "total_gb": round(usage.total / (1024**3), 2),
        }
        if free_gb < min_free_gb:
            report["errors"].append(
                f"Low disk space at {path}: {free_gb:.2f} GiB free; "
                f"configured minimum is {min_free_gb:.2f} GiB."
            )

    try:
        dataset = ShardedPatchDataset(shards_dir)
        report["patch_count"] = len(dataset)
        report["shard_count"] = int(dataset.manifest["shard_name"].nunique())
        sample_indices = sorted({0, len(dataset) - 1})
        sample_shapes = []
        for index in sample_indices:
            patch, metadata = dataset[index]
            if not torch.isfinite(patch).all():
                raise ValueError(f"Sample patch at manifest row {index} contains non-finite values.")
            if patch.ndim != 4 or patch.shape[0] not in (1, 2):
                raise ValueError(
                    f"Patch at manifest row {index} has unexpected shape {tuple(patch.shape)}; "
                    "expected (1|2, Z, Y, X)."
                )
            if len(set(patch.shape[1:])) != 1:
                raise ValueError(
                    f"Patch at manifest row {index} is not cubic: shape={tuple(patch.shape)}."
                )
            sample_shapes.append(list(patch.shape))
            logger.info(
                "Validated sample patch index=%d patch_id=%s scan_id=%s",
                index,
                metadata["patch_id"],
                metadata["scan_id"],
            )
        if len({tuple(shape) for shape in sample_shapes}) > 1:
            raise ValueError(f"First/last sample patch shapes differ: {sample_shapes}.")
        report["sample_patch_shapes"] = sample_shapes
        report["checks"].append("manifest, shard headers, referenced indices, and sample patches passed")
    except Exception as exc:
        report["errors"].append(f"Patch dataset validation failed: {type(exc).__name__}: {exc}")

    if not report["errors"]:
        try:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".preflight_", dir=checkpoint_dir, delete=True):
                pass
            report["checks"].append("checkpoint directory is writable")
        except Exception as exc:
            report["errors"].append(
                f"Checkpoint directory is not writable ({checkpoint_dir}): "
                f"{type(exc).__name__}: {exc}"
            )

    report["status"] = "passed" if not report["errors"] else "failed"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check real patch shards and the training environment without training."
    )
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/real/checkpoints"))
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    report = diagnose_training_inputs(
        shards_dir=args.shards_dir,
        checkpoint_dir=args.checkpoint_dir,
        min_free_gb=args.min_free_gb,
        require_cuda=args.require_cuda,
    )
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_name(f".{args.report.name}.tmp")
        try:
            temporary.write_text(rendered + "\n", encoding="utf-8")
            json.loads(temporary.read_text(encoding="utf-8"))
            os.replace(temporary, args.report)
        finally:
            temporary.unlink(missing_ok=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
