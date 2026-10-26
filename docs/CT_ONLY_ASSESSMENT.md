# Inspiratory CT-only COPD research assessment

This tool adds a CT-only quantitative path that does not require expiratory scans. It is intended for research and quality-controlled feature extraction, not autonomous diagnosis.

## What it computes

For the whole lung and each of the five lobes, `scripts/ct_only_assessment.py` reports:
- lung volume estimated from the segmentation mask and voxel spacing;
- mean, median, 15th percentile, and standard deviation of CT attenuation (HU);
- low-attenuation area fractions at ≤ -950 HU and ≤ -910 HU.

LAA-950 is a widely used quantitative emphysema measure on inspiratory CT. Its interpretation depends on CT acquisition/reconstruction, calibration, segmentation, and image quality. The tool checks that input volumes look like calibrated HU and that CT and lobe-mask geometry agree, but these checks do not replace radiology/segmentation quality control.

## Run feature extraction

Prepare calibrated inspiratory NIfTI CT volumes and matching five-lobe masks. For each CT file, the expected mask name is the CT filename stem plus `_lobes.nii.gz`. For example, `case001_insp.nii.gz` pairs with `case001_insp_lobes.nii.gz`.

```powershell
python scripts/ct_only_assessment.py `
  --ct-dir data/processed/ctrate/nifti `
  --lobe-dir data/processed/ctrate/lobes `
  --output-csv outputs/real/reports/ct_lobe_features.csv
```

The CT and lobe-mask directories must use the same naming convention. Masks must contain all labels 1-5 with the repository's convention: right upper, right middle, right lower, left upper, and left lower. Inspect mask overlays before trusting the measurements.

## Optional supervised research baseline

Provide one row per scan in a CSV containing `case_id,copd_label` and, when a patient has multiple scans/reconstructions, a `patient_id` column. Labels must be binary (0/1). Declare their provenance explicitly:

```powershell
python scripts/ct_only_assessment.py `
  --ct-dir data/processed/ctrate/nifti `
  --lobe-dir data/processed/ctrate/lobes `
  --output-csv outputs/real/reports/ct_lobe_features.csv `
  --labels-csv data/metadata/copd_labels.csv `
  --label-source report
```

Use `--label-source spirometry` only when the labels actually derive from suitable clinical spirometry and the corresponding diagnostic criteria. Report-derived labels must be reported as weak-label/report-proxy classification. The model uses a patient-grouped internal holdout, writes metrics and test predictions, and saves a baseline model. A single random holdout is only a preliminary estimate; use prespecified patient-level splits, external validation, calibration assessment, and confidence intervals for a research study.

## What the tool cannot claim

- CT-RATE report-derived labels are not equivalent to spirometry-confirmed COPD.
- Inspiratory CT alone does not provide a direct expiratory air-trapping measurement or paired-scan PRM.
- A model trained and evaluated within one cohort is not clinically validated.
- No prediction should be used for patient care without prospective/independent clinical validation, clinical oversight, and the appropriate regulatory and safety review.

For clinical validation, seek a permitted independent cohort with post-bronchodilator spirometry, patient-level identifiers, relevant covariates, and standardized CT acquisition. COPDGene is one possible research-access route and includes inspiratory/expiratory CT and spirometry; access is subject to its data-use process.
