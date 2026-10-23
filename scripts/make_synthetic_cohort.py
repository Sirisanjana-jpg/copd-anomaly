"""
Generate synthetic COPD patient cohort for multimodal experiments.

Saves output to data/processed/synthetic_cohort.parquet.
MANDATORY: Prominently banners all generated tables as SYNTHETIC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from copd_anomaly.synthetic.cohort_generator import (
    SYNTHETIC_BANNER,
    generate_synthetic_copd_cohort,
)
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("make_synthetic_cohort")


def create_synthetic_cohort_file(
    output_path: Path = Path("data/processed/synthetic_cohort.parquet"),
    n_patients: int = 500,
    seed: int = 42,
) -> Path:
    logger.info(SYNTHETIC_BANNER)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_copd_cohort(n_patients=n_patients, seed=seed)
    df.to_parquet(output_path, index=False)
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    logger.info(f"Successfully generated synthetic cohort: {output_path} ({len(df)} patients)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic COPD patient cohort.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/synthetic_cohort.parquet"),
    )
    parser.add_argument("--patients", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_synthetic_cohort_file(output_path=args.output, n_patients=args.patients, seed=args.seed)


if __name__ == "__main__":
    main()
