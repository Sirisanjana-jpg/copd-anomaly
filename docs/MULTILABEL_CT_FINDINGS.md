# Multi-label chest CT finding classification

The repository now includes `scripts/train_multilabel_head.py`, which trains an exploratory classifier for the 18 findings represented in CT-RATE's official multi-abnormality labels. CT-RATE includes 25,692 CT scans / 50,188 reconstructions from 21,304 patients, paired with radiology reports and report-derived labels. See the [official CT-RATE dataset page](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) and the associated [Nature Biomedical Engineering paper](https://www.nature.com/articles/s41551-025-01599-y).

## 18 supported findings

Medical material; arterial wall calcification; cardiomegaly; pericardial effusion; coronary artery wall calcification; hiatal hernia; lymphadenopathy; emphysema; atelectasis; lung nodule; lung opacity; pulmonary fibrotic sequela; pleural effusion; mosaic attenuation pattern; peribronchial thickening; consolidation; bronchiectasis; and interlobular septal thickening.

## Model input contract

The classifier consumes **image-derived embeddings**, not raw report text. Prepare one row per CT volume in an embeddings CSV, with a scan identifier column (`VolumeName`, `volume_name`, `scan_id`, or `case_id`) and numeric embedding columns such as `embedding_0`, `embedding_1`, etc. Aggregate patch embeddings to scan-level vectors before fitting the head. The labels CSV must have the same scan identifier and the exact 18 label columns used by the official CT-RATE label file.

Example:

```powershell
python scripts/train_multilabel_head.py `
  --embeddings-csv data/processed/ctrate/scan_embeddings.csv `
  --labels-csv data/processed/ctrate/train_predicted_labels.csv `
  --output-dir outputs/real/ct_rate_multilabel
```

The output contains a serialized model head, per-volume holdout probabilities, and per-label AUROC/AUPRC metrics. The script performs a patient-grouped exploratory holdout to avoid placing repeated reconstructions from the same patient in both splits. For final evaluation, use the official patient-disjoint validation cohort, and report uncertainty intervals and per-class results. Do not tune on the final validation cohort.

## Important distinction

The CT-RATE labels are automatically extracted from radiology reports. They are useful weak labels for developing multi-label CT finding models, but they are not perfect ground truth or independently verified diagnoses. A positive prediction means the model estimates that a finding may be present; it does not confirm a disease. The classifier predicts scan-level findings only and does not localize or segment lesions. Localization requires separate spatial annotations or validated localization methods.

The model head requires a real scan-level image embedding CSV. It does **not** create those embeddings from raw CT files by itself. Use a validated 3D image encoder and a preprocessing pipeline that preserves the correct scan IDs; do not create embeddings from reports or clinical label columns. No clinical validation is claimed.
