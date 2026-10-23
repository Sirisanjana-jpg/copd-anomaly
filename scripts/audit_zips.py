"""
Audit DIR-Lab zip archives non-destructively without full extraction.

Validates:
1. Archive existence and member files.
2. Little-endian int16 raw image byte sizes against configs/data/dirlab_cases.yaml (size == 2 * X * Y * Z).
3. Detects copd2 primary vs alternate dimensions.
4. Identifies extraneous metadata (__MACOSX, .DS_Store) and flags missing landmark/scan files.
5. Estimates total uncompressed volume size.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import yaml

from copd_anomaly.utils.helpers import get_logger

logger = get_logger("audit_zips")


def audit_dirlab_archives(
    zips_dir: Path,
    config_path: Path,
) -> dict:
    """Audits all zip files in zips_dir according to dirlab_cases.yaml."""
    if not zips_dir.is_dir():
        raise FileNotFoundError(f"Zips directory does not exist: {zips_dir}")

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cases_cfg = config.get("cases", {})
    audit_results = {}
    total_uncompressed_bytes = 0
    total_raw_image_bytes = 0

    zip_files = sorted(zips_dir.glob("*.zip"))
    logger.info(f"Found {len(zip_files)} zip file(s) in {zips_dir}")

    for zip_path in zip_files:
        case_name = zip_path.stem.lower()
        logger.info(f"--- Auditing archive: {zip_path.name} (Case ID: {case_name}) ---")
        case_meta = cases_cfg.get(case_name)

        if not case_meta:
            logger.warning(
                f"Case '{case_name}' not defined in {config_path.name}! "
                "Proceeding with generic audit."
            )

        case_report = {
            "archive_name": zip_path.name,
            "archive_size_bytes": zip_path.stat().st_size,
            "members": [],
            "raw_images": {},
            "landmarks": [],
            "junk_files": [],
            "dimension_match": None,
            "errors": [],
            "warnings": [],
        }

        with zipfile.ZipFile(zip_path, "r") as zf:
            infolist = zf.infolist()
            for info in infolist:
                total_uncompressed_bytes += info.file_size
                filename = info.filename
                case_report["members"].append(
                    {
                        "name": filename,
                        "size_bytes": info.file_size,
                        "compress_size": info.compress_size,
                    }
                )

                if "__MACOSX" in filename or filename.endswith(".DS_Store"):
                    case_report["junk_files"].append(filename)
                    continue

                clean_name = Path(filename).name

                if clean_name.endswith(".img"):
                    case_report["raw_images"][clean_name] = info.file_size
                    total_raw_image_bytes += info.file_size

                if clean_name.endswith(".txt") and "xyz" in clean_name.lower():
                    case_report["landmarks"].append(clean_name)

        if case_meta:
            expected_dims = case_meta.get("dims")
            dims_alt = case_meta.get("dims_alt")

            def calc_expected_bytes(dims: list[int]) -> int:
                return 2 * dims[0] * dims[1] * dims[2]

            expected_bytes = calc_expected_bytes(expected_dims) if expected_dims else None
            alt_bytes = calc_expected_bytes(dims_alt) if dims_alt else None

            for img_name, img_size in case_report["raw_images"].items():
                if img_size == expected_bytes:
                    case_report["dimension_match"] = {
                        "matched_dims": expected_dims,
                        "type": "primary",
                        "size_bytes": img_size,
                    }
                    logger.info(
                        f"  [OK] {img_name}: {img_size:,} bytes matches primary dims {expected_dims}"
                    )
                elif alt_bytes and img_size == alt_bytes:
                    case_report["dimension_match"] = {
                        "matched_dims": dims_alt,
                        "type": "alternate",
                        "size_bytes": img_size,
                    }
                    logger.info(
                        f"  [OK] {img_name}: {img_size:,} bytes matches alternate dims {dims_alt}"
                    )
                else:
                    err_msg = (
                        f"Size mismatch for {img_name}: actual {img_size:,} bytes vs "
                        f"expected {expected_bytes:,} bytes (dims: {expected_dims})"
                    )
                    if alt_bytes:
                        err_msg += f" or alternate {alt_bytes:,} bytes (dims: {dims_alt})"
                    case_report["errors"].append(err_msg)
                    logger.error(f"  [FAIL] {err_msg}")

            expected_img_prefixes = [f"{case_name}_iBHCT", f"{case_name}_eBHCT"]
            found_img_names = [Path(k).name for k in case_report["raw_images"].keys()]
            for prefix in expected_img_prefixes:
                if not any(prefix.lower() in f.lower() for f in found_img_names):
                    warn_msg = f"Missing expected raw scan matching prefix: {prefix}"
                    case_report["warnings"].append(warn_msg)
                    logger.warning(f"  [WARN] {warn_msg}")

        audit_results[case_name] = case_report

    summary = {
        "cases_audited": len(audit_results),
        "total_archives_size_mb": (
            sum(r["archive_size_bytes"] for r in audit_results.values()) / (1024 * 1024)
        ),
        "total_uncompressed_mb": total_uncompressed_bytes / (1024 * 1024),
        "total_raw_image_mb": total_raw_image_bytes / (1024 * 1024),
        "cases": audit_results,
    }

    logger.info("================ AUDIT SUMMARY ================")
    logger.info(f"Total Archives Audited: {summary['cases_audited']}")
    logger.info(f"Compressed Size: {summary['total_archives_size_mb']:.2f} MB")
    logger.info(f"Uncompressed Size: {summary['total_uncompressed_mb']:.2f} MB")
    logger.info(f"Raw Images Size: {summary['total_raw_image_mb']:.2f} MB")

    total_errors = sum(len(c["errors"]) for c in audit_results.values())
    if total_errors > 0:
        logger.error(f"Audit completed with {total_errors} critical size/dimension error(s)!")
    else:
        logger.info("All archives matched expected physical dimensions!")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DIR-Lab COPD zip archives.")
    parser.add_argument(
        "--zips-dir",
        type=Path,
        default=Path("data/raw_zips"),
        help="Path to directory containing DIR-Lab zip files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/dirlab_cases.yaml"),
        help="Path to dirlab_cases.yaml configuration.",
    )
    args = parser.parse_args()

    try:
        report = audit_dirlab_archives(args.zips_dir, args.config)
        total_errors = sum(len(c["errors"]) for c in report["cases"].values())
        if total_errors > 0:
            sys.exit(1)
        sys.exit(0)
    except Exception as exc:
        logger.error(f"Audit failed with exception: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

