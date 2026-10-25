"""
Resumable CT-RATE streaming and patch-shard preparation.

This command requires approved CT-RATE access and metadata CSVs. It stores only
patch shards/manifests and logs; raw downloaded volumes are deleted by CTRATEStreamer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from copd_anomaly.data.ctrate_streamer import CTRATEStreamer, create_tiered_splits
from copd_anomaly.patches.dataset import ShardedPatchWriter
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("stream_ctrate")


def _load_resume_writer(output_dir: Path, shard_size: int, resume: bool) -> tuple[ShardedPatchWriter, set[str]]:
    manifest_path = output_dir / "patch_manifest.parquet"
    csv_path = output_dir / "patch_manifest.csv"
    existing_manifest = manifest_path if manifest_path.is_file() else csv_path if csv_path.is_file() else None
    existing_shards = list(output_dir.glob("shard_*.npy"))
    if existing_manifest is None and existing_shards:
        raise FileNotFoundError(
            f"Found shard files in {output_dir} without a manifest. Restore the manifest or choose a fresh directory."
        )
    if existing_manifest is not None and not resume:
        raise FileExistsError(
            f"Existing patch manifest found at {existing_manifest}; use --resume or choose a fresh output directory."
        )

    writer = ShardedPatchWriter(output_dir=output_dir, shard_size=shard_size)
    completed_scans: set[str] = set()
    if existing_manifest is not None:
        existing = pd.read_parquet(manifest_path) if manifest_path.is_file() else pd.read_csv(csv_path)
        if not existing.empty:
            required = {"scan_id", "shard_name", "index_in_shard"}
            missing = required - set(existing.columns)
            if missing:
                raise ValueError(f"Existing patch manifest is missing required columns: {sorted(missing)}")
            missing_shards = [
                name for name in existing["shard_name"].astype(str).unique()
                if not (output_dir / name).is_file()
            ]
            if missing_shards:
                raise FileNotFoundError(
                    f"Manifest references missing shard files under {output_dir}: {missing_shards[:5]}"
                )
            writer.all_manifest_records = existing.to_dict(orient="records")
            completed_scans = set(existing["scan_id"].astype(str).unique())
        shard_indices = [
            int(path.stem.split("_")[-1])
            for path in output_dir.glob("shard_*.npy")
            if path.stem.split("_")[-1].isdigit()
        ]
        writer.current_shard_idx = max(shard_indices, default=-1) + 1
        logger.info("Resuming %s with %d existing patches across %d shards.", output_dir, len(writer.all_manifest_records), writer.current_shard_idx)
    return writer, completed_scans


def _append_log(log_path: Path, record: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([record])
    frame.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def stream_ctrate(
    train_metadata_path: Path,
    valid_metadata_path: Path,
    output_dir: Path,
    tier: str = "lean",
    repo_id: str = "ibrahimhamamci/CT-RATE",
    patch_size: int = 50,
    max_patches_per_scan: int = 100,
    shard_size: int = 1000,
    seed: int = 42,
    temp_dir: Path = Path("data/interim/ctrate_temp"),
    max_volumes_per_split: int | None = None,
    resume: bool = True,
) -> dict[str, int]:
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HF_READ")):
        raise EnvironmentError(
            "CT-RATE access token not found. Set HF_TOKEN (recommended) or HF_READ in your environment; "
            "never paste the token into source code or commit it."
        )
    for path in (train_metadata_path, valid_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"CT-RATE metadata CSV not found: {path}")

    train_metadata = pd.read_csv(train_metadata_path)
    valid_metadata = pd.read_csv(valid_metadata_path)
    for name, frame in (("train", train_metadata), ("valid", valid_metadata)):
        if "VolumeName" not in frame.columns:
            raise ValueError(f"{name} metadata must contain a 'VolumeName' column.")
        if frame.empty:
            raise ValueError(f"{name} metadata is empty.")

    splits = create_tiered_splits(train_metadata, valid_metadata, tier=tier, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_summary: dict[str, int] = {}
    streamer = CTRATEStreamer(
        repo_id=repo_id,
        temp_dir=temp_dir,
        patch_size=patch_size,
        max_patches_per_scan=max_patches_per_scan,
    )
    config = {
        "repo_id": repo_id,
        "tier": tier,
        "seed": seed,
        "patch_size": patch_size,
        "max_patches_per_scan": max_patches_per_scan,
        "shard_size": shard_size,
        "source_train_metadata": str(train_metadata_path),
        "source_valid_metadata": str(valid_metadata_path),
        "note": "CT-RATE report labels are proxy labels; this script does not establish clinical normality.",
    }
    (output_dir / "stream_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for split_name, split_frame in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_frame.to_csv(split_dir / "selected_volumes.csv", index=False)
        (split_dir / "selected_patient_ids.txt").write_text(
            "\n".join(sorted(split_frame["patient_id"].astype(str).unique())) + "\n",
            encoding="utf-8",
        )
        writer, completed_scans = _load_resume_writer(split_dir, shard_size, resume)
        log_path = split_dir / "stream_log.csv"
        prior_log = pd.read_csv(log_path) if log_path.is_file() else pd.DataFrame()
        completed_without_patches = set()
        if not prior_log.empty and {"scan_id", "status"}.issubset(prior_log.columns):
            completed_without_patches = set(
                prior_log.loc[prior_log["status"] == "no_patches", "scan_id"].astype(str)
            )
        volumes = split_frame.to_dict(orient="records")
        if max_volumes_per_split is not None:
            if max_volumes_per_split < 1:
                raise ValueError("max_volumes_per_split must be positive when supplied.")
            volumes = volumes[:max_volumes_per_split]

        for row in volumes:
            volume_name = str(row["VolumeName"])
            scan_id = Path(volume_name).name.replace(".nii.gz", "")
            patient_id = str(row["patient_id"])
            if scan_id in completed_scans or scan_id in completed_without_patches:
                logger.info("Skipping completed volume %s (resume).", volume_name)
                continue
            try:
                count = streamer.stream_and_process_scan(volume_name, patient_id, writer)
                if count > 0:
                    writer.close()  # Persist the current shard and full manifest after every scan.
                    completed_scans.add(scan_id)
                    status = "success"
                else:
                    status = "no_patches"
                _append_log(
                    log_path,
                    {"scan_id": scan_id, "patient_id": patient_id, "volume_name": volume_name,
                     "status": status, "patch_count": count, "error": ""},
                )
                logger.info("%s: status=%s patches=%d", volume_name, status, count)
            except Exception as exc:
                _append_log(
                    log_path,
                    {"scan_id": scan_id, "patient_id": patient_id, "volume_name": volume_name,
                     "status": "failed", "patch_count": 0, "error": str(exc)},
                )
                logger.exception("Failed volume %s; continuing to the next selected volume.", volume_name)

        # Finalize any remaining buffer and persist the manifest if at least one patch exists.
        writer.close()
        manifest = split_dir / "patch_manifest.parquet"
        if manifest.is_file():
            frame = pd.read_parquet(manifest)
            split_summary[split_name] = int(len(frame))
        else:
            split_summary[split_name] = 0
        logger.info("Split %s complete: %d patch records.", split_name, split_summary[split_name])

    if split_summary.get("ctrate_train", 0) == 0:
        raise RuntimeError(
            "No training patches were produced. Check the access token, CT-RATE file names, "
            "segmentation output, and patch-size/lung-fraction settings."
        )
    (output_dir / "patch_counts.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")
    return split_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream CT-RATE volumes into resumable patch shards.")
    parser.add_argument("--train-metadata", type=Path, required=True)
    parser.add_argument("--valid-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/ctrate_shards"))
    parser.add_argument("--tier", choices=["lean", "medium", "full"], default="lean")
    parser.add_argument("--repo-id", default="ibrahimhamamci/CT-RATE")
    parser.add_argument("--patch-size", type=int, default=50)
    parser.add_argument("--max-patches-per-scan", type=int, default=100)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temp-dir", type=Path, default=Path("data/interim/ctrate_temp"))
    parser.add_argument("--max-volumes-per-split", type=int, default=None, help="Optional capped real-data smoke run")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    summary = stream_ctrate(
        train_metadata_path=args.train_metadata,
        valid_metadata_path=args.valid_metadata,
        output_dir=args.output_dir,
        tier=args.tier,
        repo_id=args.repo_id,
        patch_size=args.patch_size,
        max_patches_per_scan=args.max_patches_per_scan,
        shard_size=args.shard_size,
        seed=args.seed,
        temp_dir=args.temp_dir,
        max_volumes_per_split=args.max_volumes_per_split,
        resume=args.resume,
    )
    logger.info("Patch extraction summary: %s", summary)


if __name__ == "__main__":
    main()
