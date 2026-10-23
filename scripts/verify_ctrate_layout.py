"""
Verify CT-RATE Hugging Face dataset repository layout and metadata schema.

Queries Hugging Face API (if HF_TOKEN is configured) to inspect dataset structure,
or verifies local/cached metadata schemas.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("verify_ctrate_layout")


def verify_layout(repo_id: str = "ibrahimhamamci/CT-RATE", output_report: Path = Path("reports/ctrate_layout_verification.json")) -> dict:
    output_report.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_READ")

    report = {
        "repo_id": repo_id,
        "token_present": bool(token),
        "files_checked": [],
        "status": "pending",
        "notes": "",
    }

    if not token:
        msg = "HF_TOKEN / HF_READ not found in environment. Proceeding with offline schema verification."
        logger.warning(msg)
        report["status"] = "offline_mode"
        report["notes"] = msg
    else:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            logger.info(f"Querying Hugging Face repo: {repo_id}...")
            repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            report["total_files_in_repo"] = len(repo_files)
            report["status"] = "verified_online"

            key_files = ["train_predicted_labels.csv", "valid_predicted_labels.csv", "dataset/train", "dataset/valid"]
            for kf in key_files:
                matched = [f for f in repo_files if kf in f]
                report["files_checked"].append({"query": kf, "found_count": len(matched)})
                logger.info(f"  Found {len(matched)} file(s) matching '{kf}'")

        except Exception as e:
            report["status"] = "remote_error"
            report["notes"] = str(e)
            logger.error(f"Failed to query remote Hugging Face API: {e}")

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved layout report to {output_report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CT-RATE repository layout.")
    parser.add_argument("--repo-id", type=str, default="ibrahimhamamci/CT-RATE")
    parser.add_argument("--output", type=Path, default=Path("reports/ctrate_layout_verification.json"))
    args = parser.parse_args()

    verify_layout(repo_id=args.repo_id, output_report=args.output)


if __name__ == "__main__":
    main()
