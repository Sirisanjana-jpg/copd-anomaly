"""
Sharded 3D Patch Dataset and Storage Engine.

Enables efficient out-of-core sharded storage and streaming PyTorch Dataset access
for millions of 3D thoracic CT patches without RAM exhaustion.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from copd_anomaly.interfaces import PatchRecord
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("patch_dataset")


class ShardedPatchWriter:
    """
    Writes extracted 3D patches into compact uint8 shards of fixed capacity.
    """

    def __init__(
        self,
        output_dir: str | Path,
        shard_size: int = 1000,
    ) -> None:
        self.output_dir = Path(output_dir)
        if shard_size < 1:
            raise ValueError(f"shard_size must be >= 1, got {shard_size}.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.current_shard_idx = 0
        self.buffer_patches: list[np.ndarray] = []
        self.buffer_records: list[dict[str, Any]] = []
        self._patch_shape: tuple[int, ...] | None = None
        self.all_manifest_records: list[dict[str, Any]] = []

    def add_patches(self, patches: np.ndarray, records: list[PatchRecord]) -> None:
        """
        Adds patches (N, C, P, P, P) and corresponding records to the writer.
        """
        if len(patches) != len(records):
            raise ValueError(
                f"Patch/record count mismatch: {len(patches)} patches but {len(records)} records."
            )
        if patches.ndim != 5:
            raise ValueError(
                f"Expected patches shaped (N, C, Z, Y, X), got array shape {patches.shape}."
            )
        if not np.issubdtype(patches.dtype, np.number) or not np.isfinite(patches).all():
            raise ValueError("Patch array contains non-numeric or non-finite values.")
        if patches.dtype != np.uint8:
            raise ValueError(f"Patch shards must be uint8; received {patches.dtype}.")
        if len(patches) == 0:
            return
        patch_shape = tuple(patches.shape[1:])
        if self._patch_shape is not None and patch_shape != self._patch_shape:
            raise ValueError(
                f"Patch shape changed within this shard set: {patch_shape} != {self._patch_shape}."
            )
        self._patch_shape = patch_shape
        for i in range(len(patches)):
            if tuple(patches[i].shape) != patch_shape:
                raise ValueError(f"Inconsistent patch shape at index {i}: {patches[i].shape}.")
            self.buffer_patches.append(patches[i])
            rec_dict = {
                "patch_id": records[i].patch_id,
                "patient_id": records[i].patient_id,
                "scan_id": records[i].scan_id,
                "center_z": records[i].center_zyx[0],
                "center_y": records[i].center_zyx[1],
                "center_x": records[i].center_zyx[2],
                "lobe_id": records[i].lobe_id if records[i].lobe_id is not None else -1,
                "lung_fraction": records[i].lung_fraction,
                "volume_weight": records[i].volume_weight,
                "laa950_fraction": records[i].laa950_fraction,
                "shard_name": f"shard_{self.current_shard_idx:04d}.npy",
                "index_in_shard": len(self.buffer_patches) - 1,
            }
            self.buffer_records.append(rec_dict)

            if len(self.buffer_patches) >= self.shard_size:
                self._flush_shard()

    def _flush_shard(self) -> None:
        if not self.buffer_patches:
            return
        shard_name = f"shard_{self.current_shard_idx:04d}.npy"
        shard_path = self.output_dir / shard_name
        arr = np.stack(self.buffer_patches, axis=0)  # Shape (M, C, P, P, P) uint8
        temp_path = shard_path.with_name(f".{shard_path.name}.tmp")
        try:
            with temp_path.open("wb") as handle:
                np.save(handle, arr, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            # Re-open before publishing so a failed/partial write is never treated as a shard.
            check = np.load(temp_path, mmap_mode="r", allow_pickle=False)
            if check.shape != arr.shape or check.dtype != np.uint8:
                raise IOError(
                    f"Shard verification failed for {shard_name}: "
                    f"shape={check.shape}, dtype={check.dtype}."
                )
            os.replace(temp_path, shard_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
        logger.info(
            f"Saved {shard_name} with {len(arr)} patches ({arr.nbytes / (1024 * 1024):.1f} MB)"
        )

        self.all_manifest_records.extend(self.buffer_records)
        self.current_shard_idx += 1
        self.buffer_patches.clear()
        self.buffer_records.clear()

    def close(self) -> Path:
        """Flushes remaining patches and writes global index manifest."""
        self._flush_shard()
        manifest_path = self.output_dir / "patch_manifest.parquet"
        if self.all_manifest_records:
            df = pd.DataFrame(self.all_manifest_records)
            parquet_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
            csv_path = manifest_path.with_suffix(".csv")
            csv_tmp = csv_path.with_name(f".{csv_path.name}.tmp")
            try:
                df.to_parquet(parquet_tmp, index=False)
                # Verify the staged manifest before atomically publishing it.
                check = pd.read_parquet(parquet_tmp)
                required = {"patch_id", "scan_id", "shard_name", "index_in_shard"}
                missing = required - set(check.columns)
                if missing or len(check) != len(df):
                    raise IOError(
                        f"Manifest verification failed: missing={sorted(missing)}, "
                        f"rows={len(check)} expected={len(df)}."
                    )
                df.to_csv(csv_tmp, index=False)
                os.replace(parquet_tmp, manifest_path)
                os.replace(csv_tmp, csv_path)
            finally:
                parquet_tmp.unlink(missing_ok=True)
                csv_tmp.unlink(missing_ok=True)
            logger.info("Wrote patch manifest to %s (%d total patches)", manifest_path, len(df))
        return manifest_path


class ShardedPatchDataset(Dataset):
    """
    PyTorch Dataset providing memory-efficient streaming access across sharded patch files.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        normal_only: bool = False,
        normal_emphysema_cap: float = 0.01,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        max_cached_shards: int = 4,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        if max_cached_shards < 1:
            raise ValueError(f"max_cached_shards must be >= 1, got {max_cached_shards}.")
        manifest_path = self.dataset_dir / "patch_manifest.parquet"
        if not manifest_path.exists():
            manifest_csv = self.dataset_dir / "patch_manifest.csv"
            if manifest_csv.exists():
                self.manifest = pd.read_csv(manifest_csv)
            else:
                self.manifest = pd.DataFrame()
        else:
            self.manifest = pd.read_parquet(manifest_path)

        if self.manifest.empty:
            raise ValueError(
                f"No patch records found in {self.dataset_dir}. "
                "Check the patch manifest and streaming failure log."
            )
        required_columns = {
            "patch_id", "patient_id", "scan_id", "shard_name", "index_in_shard",
            "lobe_id", "lung_fraction", "laa950_fraction",
        }
        missing_columns = required_columns - set(self.manifest.columns)
        if missing_columns:
            raise ValueError(
                f"Patch manifest at {self.dataset_dir} is missing columns: {sorted(missing_columns)}."
            )
        if self.manifest[["shard_name", "index_in_shard"]].isna().any().any():
            raise ValueError("Patch manifest contains missing shard references.")
        numeric_indices = pd.to_numeric(self.manifest["index_in_shard"], errors="coerce")
        if not np.isfinite(numeric_indices.to_numpy(dtype=np.float64)).all():
            raise ValueError("Patch manifest contains non-numeric shard indices.")
        if (numeric_indices < 0).any():
            raise ValueError("Patch manifest contains negative index_in_shard values.")
        if not np.equal(numeric_indices, np.floor(numeric_indices)).all():
            raise ValueError("Patch manifest contains non-integer shard indices.")
        self.manifest["index_in_shard"] = numeric_indices.astype(np.int64)
        shard_names = self.manifest["shard_name"].astype(str).unique()
        unsafe_names = [
            name for name in shard_names if Path(name).name != name or not name.endswith(".npy")
        ]
        if unsafe_names:
            raise ValueError(
                f"Patch manifest contains unsafe shard paths: {unsafe_names[:10]}."
            )
        missing_shards = sorted(
            name for name in shard_names if not (self.dataset_dir / name).is_file()
        )
        if missing_shards:
            raise FileNotFoundError(
                f"Patch manifest references missing shard files: {missing_shards[:10]}."
            )
        if self.manifest["patch_id"].astype(str).duplicated().any():
            raise ValueError("Patch manifest contains duplicate patch_id values.")
        if self.manifest.duplicated(subset=["shard_name", "index_in_shard"]).any():
            raise ValueError("Multiple manifest rows point to the same patch in a shard.")
        self._validate_shard_references()
        if normal_only and "laa950_fraction" in self.manifest.columns:
            self.manifest = self.manifest[
                self.manifest["laa950_fraction"] <= normal_emphysema_cap
            ].reset_index(drop=True)
            logger.info(
                f"Filtered ShardedPatchDataset to normal patches (LAA-950 <= {normal_emphysema_cap}): {len(self.manifest)} remaining."
            )

        self.transform = transform
        self.max_cached_shards = max_cached_shards
        self._shard_cache: dict[str, np.ndarray] = {}
        self._cache_keys_lru: list[str] = []

    def _validate_shard_references(self) -> None:
        """Validate shard headers and manifest indices without loading patch arrays into RAM."""
        expected_patch_shape: tuple[int, ...] | None = None
        for shard_name, rows in self.manifest.groupby("shard_name", sort=False):
            shard_path = self.dataset_dir / str(shard_name)
            try:
                shard = np.load(shard_path, mmap_mode="r", allow_pickle=False)
            except Exception as exc:
                raise IOError(f"Cannot read patch shard {shard_path}: {exc}") from exc
            if shard.ndim != 5 or shard.dtype != np.uint8:
                raise ValueError(
                    f"Invalid patch shard {shard_path}: expected uint8 5D array, "
                    f"got shape={shard.shape}, dtype={shard.dtype}."
                )
            patch_shape = tuple(shard.shape[1:])
            if expected_patch_shape is None:
                expected_patch_shape = patch_shape
            elif patch_shape != expected_patch_shape:
                raise ValueError(
                    f"Patch shape differs across shards: {shard_path} has {patch_shape}, "
                    f"expected {expected_patch_shape}."
                )
            indices = rows["index_in_shard"].astype(int)
            if indices.max() >= len(shard):
                raise IndexError(
                    f"Manifest references index {indices.max()} in {shard_path}, "
                    f"but shard contains only {len(shard)} patches."
                )
            if indices.min() < 0:
                raise IndexError(f"Manifest contains a negative index for {shard_path}.")

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_shard(self, shard_name: str) -> np.ndarray:
        if shard_name in self._shard_cache:
            # Move to MRU
            self._cache_keys_lru.remove(shard_name)
            self._cache_keys_lru.append(shard_name)
            return self._shard_cache[shard_name]

        shard_path = self.dataset_dir / shard_name
        try:
            data = np.load(shard_path, mmap_mode="r", allow_pickle=False)
        except Exception as exc:
            raise IOError(f"Cannot read patch shard {shard_path}: {exc}") from exc
        if data.ndim != 5 or data.dtype != np.uint8:
            raise ValueError(
                f"Invalid patch shard {shard_path}: expected uint8 5D array, "
                f"got shape={data.shape}, dtype={data.dtype}."
            )

        if len(self._cache_keys_lru) >= self.max_cached_shards:
            oldest = self._cache_keys_lru.pop(0)
            self._shard_cache.pop(oldest, None)

        self._shard_cache[shard_name] = data
        self._cache_keys_lru.append(shard_name)
        return data

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        row = self.manifest.iloc[idx]
        shard_name = row["shard_name"]
        index_in_shard = int(row["index_in_shard"])

        shard_data = self._load_shard(str(shard_name))
        if index_in_shard < 0 or index_in_shard >= len(shard_data):
            raise IndexError(
                f"Manifest row {idx} references index {index_in_shard} in {shard_name}, "
                f"but shard length is {len(shard_data)}."
            )
        patch_uint8 = np.array(shard_data[index_in_shard])  # Shape (C, P, P, P)

        # Scale uint8 to float32 [0.0, 1.0]
        tensor = torch.from_numpy(patch_uint8).float() / 255.0

        if self.transform is not None:
            tensor = self.transform(tensor)

        meta = {
            "patch_id": row["patch_id"],
            "patient_id": row["patient_id"],
            "scan_id": row["scan_id"],
            "lobe_id": int(row["lobe_id"]),
            "lung_fraction": float(row["lung_fraction"]),
            "laa950_fraction": float(row["laa950_fraction"]),
        }
        return tensor, meta
