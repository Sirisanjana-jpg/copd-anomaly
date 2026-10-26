"""Run the DIR-Lab raw-ZIP -> NIfTI -> five-lobe patches -> model pipeline.

This is an exploratory, self-supervised pipeline for the small DIR-Lab cohort.
It does not create a held-out test set and must not be used to claim generalization.
Raw archives stay under data/ and are never committed by this script.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from nibabel.processing import resample_to_output

CASE_PATTERN = re.compile(r"(copd\d+)", re.IGNORECASE)


def discover_case_archives(data_dir: Path, cases: list[str] | None = None) -> dict[str, Path]:
    """Find one archive per configured DIR-Lab case anywhere below data_dir."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    found: dict[str, Path] = {}
    requested = {case.lower() for case in cases} if cases else None
    for path in sorted(data_dir.rglob("*.zip")):
        match = CASE_PATTERN.search(path.stem)
        if not match:
            continue
        case_id = match.group(1).lower()
        if requested is not None and case_id not in requested:
            continue
        if case_id in found:
            raise ValueError(
                f"Multiple ZIPs match {case_id}: {found[case_id]} and {path}. "
                "Keep one authoritative archive per case or pass a narrower --data-dir."
            )
        try:
            with zipfile.ZipFile(path) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise ValueError(f"Corrupt ZIP member {bad_member!r} in {path}")
                names = [Path(member.filename).name for member in archive.namelist()]
                if not any(name.endswith(".img") and "_iBHCT" in name for name in names):
                    raise ValueError(f"{path} has no DIR-Lab inspiratory *_iBHCT*.img volume.")
                if not any(name.endswith(".img") and "_eBHCT" in name for name in names):
                    raise ValueError(f"{path} has no DIR-Lab expiratory *_eBHCT*.img volume.")
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Not a valid ZIP archive: {path}") from exc
        found[case_id] = path
    if requested is not None:
        missing = sorted(requested - set(found))
        if missing:
            raise FileNotFoundError(
                f"No matching ZIP archive found under {data_dir} for {missing}."
            )
    if not found:
        raise FileNotFoundError(
            f"No DIR-Lab case ZIPs found under {data_dir}. Expected filenames containing "
            "copd1.zip ... copd10.zip (archives must include *_iBHCT*.img and *_eBHCT*.img)."
        )
    if len(found) < 2:
        raise ValueError(
            f"Found only {len(found)} case archive. At least two cases are needed for a "
            "minimal exploratory training run; this is still not a held-out evaluation."
        )
    return dict(sorted(found.items()))


def run_command(command: list[str], repo_root: Path, env: dict[str, str]) -> None:
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def resample_inspiratory_scans(nifti_dir: Path, output_dir: Path, cases: list[str]) -> int:
    """Resample inspiratory CTs to an isotropic 1 mm grid before cubic patch sampling."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for case_id in cases:
        source = nifti_dir / f"{case_id}_insp.nii.gz"
        if not source.is_file():
            raise FileNotFoundError(f"Expected inspiratory NIfTI is missing: {source}")
        image = nib.load(str(source))
        resampled = resample_to_output(image, voxel_sizes=(1.0, 1.0, 1.0), order=1)
        values = resampled.get_fdata(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Resampling produced non-finite HU values for {case_id}.")
        nib.save(resampled, str(output_dir / source.name))
        written += 1
    return written


def create_binary_masks_from_lobes(
    nifti_dir: Path, lobe_dir: Path, masks_dir: Path, cases: list[str]
) -> int:
    """Create whole-lung masks from the union of validated anatomical lobe labels."""
    masks_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for case_id in cases:
        scan = nifti_dir / f"{case_id}_insp.nii.gz"
        lobe_path = lobe_dir / f"{case_id}_lobes.nii.gz"
        if not scan.is_file():
            raise FileNotFoundError(f"Expected inspiratory NIfTI is missing: {scan}")
        if not lobe_path.is_file():
            raise FileNotFoundError(f"Expected five-lobe segmentation is missing: {lobe_path}")
        image = nib.load(str(scan))
        lobe_image = nib.load(str(lobe_path))
        if image.shape[:3] != lobe_image.shape[:3] or not np.allclose(
            image.affine, lobe_image.affine, rtol=0.0, atol=1e-4
        ):
            raise ValueError(f"CT/lobe-mask geometry mismatch for {case_id}.")
        labels = np.asarray(lobe_image.get_fdata(dtype=np.float32))
        if not np.isfinite(labels).all() or not np.equal(labels, np.round(labels)).all():
            raise ValueError(f"Non-finite or non-integer lobe labels for {case_id}.")
        present = set(np.unique(labels).astype(int)) - {0}
        if present != {1, 2, 3, 4, 5}:
            raise ValueError(f"{case_id} does not contain all five lobe IDs: {sorted(present)}.")
        mask = (labels > 0).astype(np.uint8)
        output = nib.Nifti1Image(mask, image.affine, image.header.copy())
        output.set_data_dtype(np.uint8)
        nib.save(output, str(masks_dir / f"{case_id}_insp_mask.nii.gz"))
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find DIR-Lab ZIPs in data/, extract NIfTI scans, build five-lobe patches, "
        "train a self-supervised encoder and fit a patch-density model."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/data/dirlab_cases.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/dirlab_run"))
    parser.add_argument(
        "--cases", nargs="*", default=None, help="Optional case IDs, e.g. copd1 copd2."
    )
    parser.add_argument("--patch-size", type=int, default=50)
    parser.add_argument("--max-patches", type=int, default=100)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--components", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo_root / args.data_dir
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    output_root = (
        args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Case configuration not found: {config_path}")
    if args.patch_size < 16 or args.patch_size % 2:
        raise ValueError("--patch-size must be an even integer >= 16 for the current sampler.")
    if args.max_patches < 1 or args.shard_size < 1 or args.epochs < 1:
        raise ValueError("--max-patches, --shard-size and --epochs must be positive.")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be >= 2 for contrastive training.")
    if args.components < 1:
        raise ValueError("--components must be positive.")

    with config_path.open(encoding="utf-8") as handle:
        configured_cases = set(yaml.safe_load(handle).get("cases", {}))
    archives = discover_case_archives(data_dir, args.cases)
    unknown = set(archives) - configured_cases
    if unknown:
        raise ValueError(f"ZIPs found for cases absent from {config_path}: {sorted(unknown)}")
    case_ids = list(archives)
    print(f"Found {len(archives)} validated DIR-Lab archives:")
    for case_id, path in archives.items():
        print(f"  {case_id}: {path} ({path.stat().st_size / (1024**2):.1f} MiB)")

    nifti_dir = output_root / "nifti"
    interim_dir = output_root / "interim"
    manifest_path = output_root / "dirlab_manifest.parquet"
    isotropic_dir = output_root / "nifti_1mm"
    lobe_dir = output_root / "lobes"
    masks_dir = output_root / "masks"
    shards_dir = output_root / "patches"
    checkpoint_dir = repo_root / "outputs" / "real" / "dirlab_run" / "checkpoints"
    reports_dir = repo_root / "outputs" / "real" / "dirlab_run" / "reports"
    if (shards_dir / "patch_manifest.parquet").exists() or (
        shards_dir / "patch_manifest.csv"
    ).exists():
        raise FileExistsError(
            f"Patch manifest already exists under {shards_dir}. Use a fresh --output-root "
            "rather than mixing new and old shards."
        )

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    data_dir.mkdir(parents=True, exist_ok=True)
    staging_parent = data_dir / "interim"
    staging_parent.mkdir(parents=True, exist_ok=True)

    # Existing manifest builder expects flat case-id.zip names; stage links/copies temporarily.
    with tempfile.TemporaryDirectory(prefix="dirlab_zip_index_", dir=staging_parent) as temp:
        zip_index = Path(temp)
        for case_id, source in archives.items():
            target = zip_index / f"{case_id}.zip"
            try:
                target.symlink_to(source.resolve())
            except OSError:
                shutil.copy2(source, target)
        run_command(
            [
                sys.executable,
                "scripts/build_dirlab_manifest.py",
                "--zips-dir",
                str(zip_index),
                "--config",
                str(config_path),
                "--interim-dir",
                str(interim_dir),
                "--output-nifti-dir",
                str(nifti_dir),
                "--manifest-parquet",
                str(manifest_path),
                "--cases",
                *case_ids,
            ],
            repo_root,
            env,
        )

    # Resample before patching: native DIR-Lab scans have 2.5 mm slice spacing and
    # ~0.6-0.7 mm in-plane spacing, so voxel-cubic patches would not be physically cubic.
    n_resampled = resample_inspiratory_scans(nifti_dir, isotropic_dir, case_ids)
    print(f"Resampled {n_resampled} inspiratory scans to 1 mm isotropic spacing.")

    # Five-lobe segmentation is mandatory for this project's spatial analysis; no fake fallback.
    run_command(
        [
            sys.executable,
            "scripts/segment_lobes.py",
            "--nifti-dir",
            str(isotropic_dir),
            "--output-dir",
            str(lobe_dir),
            "--cases",
            *case_ids,
        ],
        repo_root,
        env,
    )
    mask_count = create_binary_masks_from_lobes(isotropic_dir, lobe_dir, masks_dir, case_ids)
    print(f"Created {mask_count} whole-lung masks from five-lobe labels.")

    run_command(
        [
            sys.executable,
            "scripts/make_patches.py",
            "--nifti-dir",
            str(isotropic_dir),
            "--masks-dir",
            str(masks_dir),
            "--lobe-masks-dir",
            str(lobe_dir),
            "--output-dir",
            str(shards_dir),
            "--patch-size",
            str(args.patch_size),
            "--max-patches",
            str(args.max_patches),
            "--shard-size",
            str(args.shard_size),
            "--cases",
            *case_ids,
        ],
        repo_root,
        env,
    )

    manifest_csv = shards_dir / "patch_manifest.csv"
    patch_manifest = (
        pd.read_parquet(shards_dir / "patch_manifest.parquet")
        if (shards_dir / "patch_manifest.parquet").is_file()
        else pd.read_csv(manifest_csv)
    )
    if len(patch_manifest) < args.batch_size:
        raise ValueError(
            f"Only {len(patch_manifest)} patches were extracted; need at least "
            f"{args.batch_size} to train. Inspect masks and patch settings."
        )
    eligible = int(
        (pd.to_numeric(patch_manifest["laa950_fraction"], errors="coerce") <= 0.01).sum()
    )
    if eligible < 2:
        raise ValueError(
            f"Only {eligible} patches satisfy the current LAA-950 <= 0.01 pseudo-normal "
            "filter; cannot fit a density model. This cohort may not provide suitable "
            "normal-like patches. Do not silently relax the threshold."
        )
    components = min(args.components, eligible)
    print(
        f"Extracted {len(patch_manifest)} patches; {eligible} pass the LAA-950 pseudo-normal "
        f"filter. Fitting {components} GMM component(s)."
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training_command = [
        sys.executable,
        "scripts/train_encoder.py",
        "--shards-dir",
        str(shards_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--checkpoint-every-steps",
        "25",
    ]
    if args.require_cuda:
        training_command.append("--require-cuda")
    run_command(training_command, repo_root, env)

    encoder_path = checkpoint_dir / "encoder_best.pt"
    density_path = checkpoint_dir / "density_model.joblib"
    run_command(
        [
            sys.executable,
            "scripts/fit_density.py",
            "--shards-dir",
            str(shards_dir),
            "--encoder-ckpt",
            str(encoder_path),
            "--output-model",
            str(density_path),
            "--model-type",
            "gmm",
            "--components",
            str(components),
        ],
        repo_root,
        env,
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "scripts/score_anomaly.py",
            "--real-inference",
            "--shards-dir",
            str(shards_dir),
            "--encoder-ckpt",
            str(encoder_path),
            "--density-model",
            str(density_path),
            "--output-dir",
            str(reports_dir),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
        ],
        repo_root,
        env,
    )
    run_command(
        [
            sys.executable,
            "scripts/lobe_analysis.py",
            "--manifest",
            str(manifest_path),
            "--patch-scores",
            str(reports_dir / "patch_anomaly_scores.csv"),
            "--output",
            str(reports_dir / "lobe_features.csv"),
        ],
        repo_root,
        env,
    )

    print("\nDIR-Lab exploratory pipeline completed.")
    print(f"Converted scans + manifest: {output_root}")
    print(f"Patch shards: {shards_dir}")
    print(f"Encoder checkpoint: {encoder_path}")
    print(f"Density model: {density_path}")
    print(f"Patch/scan/patient scores and lobe features: {reports_dir}")
    print(
        "RESEARCH LIMITATION: DIR-Lab has only a small set of COPD cases. This run trains and "
        "scores on the same cohort, has no independent held-out evaluation, and uses "
        "LAA-950 <= 0.01 only as a pseudo-normal heuristic. Treat outputs as pipeline/QC "
        "artifacts, not validated diagnostic performance or evidence of generalization."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
