"""
Sharded 3D Patch Dataset and Storage Engine.

Enables efficient out-of-core sharded storage and streaming PyTorch Dataset access
for millions of 3D thoracic CT patches without RAM exhaustion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
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
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.current_shard_idx = 0
        self.buffer_patches: list[np.ndarray] = []
        self.buffer_records: list[dict[str, Any]] = []
        self.all_manifest_records: list[dict[str, Any]] = []

    def add_patches(self, patches: np.ndarray, records: list[PatchRecord]) -> None:
        """
        Adds patches (N, C, P, P, P) and corresponding records to the writer.
        """
        for i in range(len(patches)):
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
        np.save(shard_path, arr)
        logger.info(f"Saved {shard_name} with {len(arr)} patches ({arr.nbytes / (1024*1024):.1f} MB)")

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
            df.to_parquet(manifest_path, index=False)
            df.to_csv(manifest_path.with_suffix(".csv"), index=False)
            logger.info(f"Wrote patch manifest to {manifest_path} ({len(df)} total patches)")
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
        manifest_path = self.dataset_dir / "patch_manifest.parquet"
        if not manifest_path.exists():
            manifest_csv = self.dataset_dir / "patch_manifest.csv"
            if manifest_csv.exists():
                self.manifest = pd.read_csv(manifest_csv)
            else:
                self.manifest = pd.DataFrame()
        else:
            self.manifest = pd.read_parquet(manifest_path)

        if normal_only and not self.manifest.empty and "laa950_fraction" in self.manifest.columns:
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

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_shard(self, shard_name: str) -> np.ndarray:
        if shard_name in self._shard_cache:
            # Move to MRU
            self._cache_keys_lru.remove(shard_name)
            self._cache_keys_lru.append(shard_name)
            return self._shard_cache[shard_name]

        shard_path = self.dataset_dir / shard_name
        data = np.load(shard_path, mmap_mode="r")

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

        shard_data = self._load_shard(shard_name)
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
