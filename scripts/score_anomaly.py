"""
Evaluate report-label proxy metrics.

This script currently supports an explicit synthetic smoke test only. Real inference
must be wired to a patient/patch manifest before a real evaluation report is produced.
Synthetic results are never labeled as clinical COPD classification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from copd_anomaly.anomaly.density import GaussianMixtureDensity
from copd_anomaly.evaluation.metrics import ModelEvaluator
from copd_anomaly.models.encoder import ResNet3DEncoder
from copd_anomaly.patches.dataset import ShardedPatchDataset
from copd_anomaly.utils.helpers import get_logger

logger = get_logger("score_anomaly")
PROXY_WARNING = (
    "SYNTHETIC DRY RUN — report-label proxy, not COPD classification; not real CT evaluation."
)


def score_and_evaluate(
    density_model_path: Path,
    output_report_path: Path = Path("reports/proxy_auc_report.csv"),
    dry_run: bool = False,
    n_bootstraps: int = 500,
) -> pd.DataFrame:
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be >= 1.")
    output_parts = {part.lower() for part in output_report_path.parts}
    if dry_run and "real" in output_parts:
        raise ValueError(
            "Synthetic evaluation reports cannot be written under a real output directory."
        )
    if not dry_run and "synthetic" in output_parts:
        raise ValueError(
            "Real evaluation reports cannot be written under a synthetic output directory."
        )
    if dry_run:
        if output_report_path == Path("reports/proxy_auc_report.csv"):
            output_report_path = Path("outputs/synthetic/reports/proxy_auc_report.csv")
        logger.warning(PROXY_WARNING)
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 50 + [1] * 50)
        scores_mean = np.concatenate([rng.normal(10.0, 3.0, 50), rng.normal(18.0, 4.0, 50)])
        scores_p90 = np.concatenate([rng.normal(12.0, 3.0, 50), rng.normal(22.0, 4.0, 50)])
        df_patients = pd.DataFrame(
            {
                "patient_id": [f"synthetic_{i:03d}" for i in range(100)],
                "y_true": y_true,
                "score_mean": scores_mean,
                "score_p90": scores_p90,
            }
        )
        metric_type = PROXY_WARNING
    else:
        if not density_model_path.is_file():
            raise FileNotFoundError(
                f"Density model not found: {density_model_path}. No report was written. "
                "Fit a real density model first, or use --dry-run for synthetic smoke testing."
            )
        raise NotImplementedError(
            "Real patient-level scoring is not implemented yet: the current script has no real "
            "patient/patch inference input. Refusing to write an empty or fabricated real evaluation. "
            "Use --dry-run only to test the metrics code."
        )

    evaluator = ModelEvaluator()
    results = []
    for score_col in ("score_mean", "score_p90"):
        metrics = evaluator.evaluate_proxy_auc(
            y_true=df_patients["y_true"].values,
            y_score=df_patients[score_col].values,
            n_bootstraps=n_bootstraps,
        )
        metrics["aggregator"] = score_col
        metrics["data_status"] = metric_type
        results.append(metrics)
    report = pd.DataFrame(results)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_report_path, index=False)
    logger.info("Saved explicitly synthetic proxy smoke-test report to %s", output_report_path)
    return report


def score_real_patches(
    shards_dir: Path,
    encoder_ckpt: Path,
    density_model_path: Path,
    output_dir: Path = Path("outputs/real/reports"),
    batch_size: int = 8,
    device: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every real patch and aggregate to scan/patient rows without inventing labels."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if not shards_dir.is_dir():
        raise FileNotFoundError(f"Real patch shard directory not found: {shards_dir}")
    if not encoder_ckpt.is_file():
        raise FileNotFoundError(f"Trained real encoder checkpoint not found: {encoder_ckpt}")
    if not density_model_path.is_file():
        raise FileNotFoundError(f"Fitted real density model not found: {density_model_path}")
    if "synthetic" in {part.lower() for part in encoder_ckpt.parts}:
        raise ValueError("Synthetic encoder checkpoints cannot be used for real inference.")
    if "synthetic" in {part.lower() for part in density_model_path.parts}:
        raise ValueError("Synthetic density models cannot be used for real inference.")
    if "synthetic" in {part.lower() for part in output_dir.parts}:
        raise ValueError("Real inference outputs cannot be written under a synthetic directory.")

    dataset = ShardedPatchDataset(shards_dir)
    device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for inference but is not available.")
    try:
        checkpoint = torch.load(encoder_ckpt, map_location=device_obj, weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Unable to load encoder checkpoint {encoder_ckpt}: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("training_mode") != "real_patch_training":
        raise ValueError("Inference requires a checkpoint marked real_patch_training.")
    if checkpoint.get("epoch_complete", True) is False or int(checkpoint.get("trained_steps", 0)) < 1:
        raise ValueError("Inference requires a completed, trained encoder checkpoint.")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Encoder checkpoint is missing model_state_dict.")
    in_channels = int(checkpoint.get("in_channels", 1))
    sample, _ = dataset[0]
    if int(sample.shape[0]) != in_channels:
        raise ValueError(
            f"Checkpoint expects {in_channels} channel(s), but shards contain {sample.shape[0]}."
        )
    encoder = ResNet3DEncoder(
        in_channels=in_channels, base_filters=16, embedding_dim=128, projection_dim=64
    ).to(device_obj)
    encoder.load_state_dict(checkpoint["model_state_dict"])
    encoder.eval()
    density = GaussianMixtureDensity.load(str(density_model_path))
    if density.training_mode != "real_patch_training":
        raise ValueError(
            "Real inference requires a density model with explicit real_patch_training provenance. "
            "Refit density from the real training split; legacy/unverified artifacts are rejected."
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for patches, _metadata in loader:
            # Density fitting uses the contrastive projection (second model output).
            _embedding, projection = encoder(patches.to(device_obj, dtype=torch.float32))
            embeddings.append(projection.detach().cpu().numpy())
    if not embeddings:
        raise ValueError("No real patch embeddings were produced.")
    z = np.concatenate(embeddings, axis=0)
    if len(z) != len(dataset.manifest) or not np.isfinite(z).all():
        raise ValueError("Embedding count/non-finite check failed; refusing to write scores.")
    scores = -density.log_prob(z)
    if scores.shape != (len(dataset.manifest),) or not np.isfinite(scores).all():
        raise ValueError("Density model returned invalid anomaly scores.")

    output_dir.mkdir(parents=True, exist_ok=True)
    patch_scores = dataset.manifest.copy().reset_index(drop=True)
    patch_scores["anomaly_score"] = scores.astype(float)
    patch_scores["data_status"] = "REAL_CT_PATCH_INFERENCE_UNVALIDATED"
    patch_path = output_dir / "patch_anomaly_scores.csv"
    patch_tmp = patch_path.with_suffix(".csv.tmp")
    patch_scores.to_csv(patch_tmp, index=False)
    pd.read_csv(patch_tmp, nrows=1)
    patch_tmp.replace(patch_path)

    required = {"patient_id", "scan_id", "lobe_id", "anomaly_score"}
    if not required.issubset(patch_scores.columns):
        raise ValueError(f"Scored manifest is missing columns: {sorted(required - set(patch_scores.columns))}")
    scan_records = []
    for (patient_id, scan_id), group in patch_scores.groupby(["patient_id", "scan_id"], sort=True):
        row = {
            "patient_id": str(patient_id),
            "scan_id": str(scan_id),
            "patch_count": int(len(group)),
            "anomaly_mean": float(group["anomaly_score"].mean()),
            "anomaly_p90": float(group["anomaly_score"].quantile(0.90)),
            "anomaly_p95": float(group["anomaly_score"].quantile(0.95)),
            "anomaly_max": float(group["anomaly_score"].max()),
            "data_status": "REAL_CT_PATCH_INFERENCE_UNVALIDATED",
        }
        for lobe_id, name in {1: "rul", 2: "rml", 3: "rll", 4: "lul", 5: "lll"}.items():
            lobe_scores = group.loc[group["lobe_id"] == lobe_id, "anomaly_score"]
            row[f"{name}_patch_count"] = int(len(lobe_scores))
            row[f"{name}_anomaly_mean"] = float(lobe_scores.mean()) if len(lobe_scores) else float("nan")
            row[f"{name}_anomaly_p90"] = float(lobe_scores.quantile(0.90)) if len(lobe_scores) else float("nan")
        row["lobes_observed_count"] = sum(row[f"{name}_patch_count"] > 0 for name in ("rul", "rml", "rll", "lul", "lll"))
        scan_records.append(row)
    scan_scores = pd.DataFrame(scan_records)
    patient_scores = (
        scan_scores.groupby("patient_id", as_index=False)
        .agg(
            scan_count=("scan_id", "nunique"),
            patch_count=("patch_count", "sum"),
            anomaly_mean=("anomaly_mean", "mean"),
            anomaly_p90=("anomaly_p90", "mean"),
            anomaly_p95=("anomaly_p95", "mean"),
            anomaly_max=("anomaly_max", "max"),
        )
    )
    patient_scores["data_status"] = "REAL_CT_PATCH_INFERENCE_UNVALIDATED"
    for name in ("rul", "rml", "rll", "lul", "lll"):
        patient_scores[f"{name}_anomaly_mean"] = scan_scores.groupby("patient_id")[f"{name}_anomaly_mean"].mean().reindex(patient_scores["patient_id"]).to_numpy()
        patient_scores[f"{name}_anomaly_p90"] = scan_scores.groupby("patient_id")[f"{name}_anomaly_p90"].mean().reindex(patient_scores["patient_id"]).to_numpy()
        patient_scores[f"{name}_patch_count"] = scan_scores.groupby("patient_id")[f"{name}_patch_count"].sum().reindex(patient_scores["patient_id"]).to_numpy()

    for frame, filename in (
        (scan_scores, "scan_anomaly_scores.csv"),
        (patient_scores, "patient_anomaly_scores.csv"),
    ):
        path = output_dir / filename
        temp = path.with_suffix(".csv.tmp")
        frame.to_csv(temp, index=False)
        pd.read_csv(temp, nrows=1)
        temp.replace(path)
    logger.info(
        "Scored %d real patches from %d scans and %d patients. Scores are unvalidated research outputs.",
        len(patch_scores), len(scan_scores), len(patient_scores)
    )
    return patch_scores, scan_scores, patient_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate report-label proxy metrics.")
    parser.add_argument(
        "--density-model", type=Path, default=Path("outputs/real/checkpoints/density_model.joblib")
    )
    parser.add_argument("--output-report", type=Path, default=Path("reports/proxy_auc_report.csv"))
    parser.add_argument("--dry-run", action="store_true", help="Evaluate synthetic scores only")
    parser.add_argument("--bootstraps", type=int, default=500)
    parser.add_argument(
        "--real-inference",
        action="store_true",
        help="Run real patch inference and write patch/scan/patient scores; no clinical labels are inferred.",
    )
    parser.add_argument("--shards-dir", type=Path, default=Path("data/processed/ctrate_lean/test"))
    parser.add_argument("--encoder-ckpt", type=Path, default=Path("outputs/real/checkpoints/encoder_best.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/real/reports"))
    args = parser.parse_args()
    if args.real_inference:
        score_real_patches(
            shards_dir=args.shards_dir,
            encoder_ckpt=args.encoder_ckpt,
            density_model_path=args.density_model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        score_and_evaluate(
            density_model_path=args.density_model,
            output_report_path=args.output_report,
            dry_run=args.dry_run,
            n_bootstraps=args.bootstraps,
        )


if __name__ == "__main__":
    main()
