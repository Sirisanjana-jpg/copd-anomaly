from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.train_multilabel_head import CTRATE_LABELS, train_multilabel_head


def test_train_multilabel_head_creates_metrics_and_predictions(tmp_path: Path) -> None:
    rng = np.random.default_rng(10)
    embeddings = []
    labels = []
    for patient in range(120):
        volume_name = f"train_{patient}_a_1"
        y = [int((patient + idx) % (idx + 3) == 0) for idx in range(len(CTRATE_LABELS))]
        # Ensure each label has both classes and signal in the features.
        embeddings.append({
            "VolumeName": volume_name,
            "embedding_0": float(y[7] * 2 + rng.normal()),
            "embedding_1": float(y[9] * 2 + rng.normal()),
            "embedding_2": float(rng.normal()),
        })
        labels.append({"VolumeName": volume_name, **dict(zip(CTRATE_LABELS, y, strict=True))})
    emb_path = tmp_path / "embeddings.csv"
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(embeddings).to_csv(emb_path, index=False)
    pd.DataFrame(labels).to_csv(labels_path, index=False)

    metrics = train_multilabel_head(emb_path, labels_path, tmp_path / "out", seed=5)
    assert metrics["label_source"] == "CT-RATE report-derived pseudo-labels"
    assert metrics["clinical_validation_status"] == "NOT_CLINICALLY_VALIDATED"
    assert len(metrics["per_label_metrics"]) == 18
    assert (tmp_path / "out" / "metrics.json").is_file()
    assert (tmp_path / "out" / "holdout_predictions.csv").is_file()
    assert (tmp_path / "out" / "ct_rate_multilabel_head.joblib").is_file()


def test_train_multilabel_head_rejects_missing_label_columns(tmp_path: Path) -> None:
    emb_path = tmp_path / "embeddings.csv"
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame({"scan_id": ["train_1_a_1"], "embedding_0": [0.1]}).to_csv(
        emb_path, index=False
    )
    pd.DataFrame({"scan_id": ["train_1_a_1"]}).to_csv(labels_path, index=False)
    with pytest.raises(ValueError, match="missing CT-RATE finding columns"):
        train_multilabel_head(emb_path, labels_path, tmp_path / "out")
