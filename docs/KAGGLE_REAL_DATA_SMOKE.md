# Kaggle real-data smoke validation

This is an operational checklist, not evidence that any real-data run has completed. Keep CT-RATE source files and outputs private, and comply with the dataset's access terms. Do not place access tokens in notebook cells, source files, outputs, or Git.

## Important: Kaggle and Google Drive

Kaggle notebooks do not provide Colab's native `drive.mount()` workflow. Files stored in Google Drive are not automatically visible under `/kaggle/input`.

Preferred choices:
1. If CT-RATE is already stored in an approved private Kaggle input and its terms permit that storage, use its `/kaggle/input/<dataset-name>/...` path.
2. If the only accessible copy is in Google Drive, use a runtime that supports Drive mounting (for example, Colab) rather than making the data publicly shareable or copying gated data to an unapproved service.
3. Do not publish a public Kaggle Dataset containing CT-RATE or DIR-Lab data. Do not upload raw scans, masks, landmarks, patch shards, or derived artifacts to this repository.

## 1. Start a fresh Kaggle notebook

- Enable GPU in Notebook Settings.
- Enable Internet only if permitted by the dataset terms and required for installing packages or accessing the approved dataset source.
- Add only approved, private data inputs. Do not store tokens in notebook code.
- Confirm the runtime has enough free disk before downloading or preprocessing scans.

Run:

```python
import os, shutil, sys, torch

print("Python:", sys.version)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("Disk free GiB:", round(shutil.disk_usage("/kaggle/working").free / 1024**3, 2))
print("Kaggle inputs:", os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "missing")
```

Stop if CUDA is required for training but unavailable. The first data-stream smoke test can run on CPU if it fits within the session's resources; do not start a long encoder run yet.

## 2. Clone the remediation branch and install

```bash
%cd /kaggle/working
!git clone --branch remediation/phase0-safety-ci --single-branch https://github.com/Sirisanjana-jpg/copd-anomaly.git
%cd /kaggle/working/copd-anomaly
!python -m pip install -U pip
!python -m pip install -e ".[dev]"
```

If dependency installation fails, preserve the complete pip error output. Do not silently upgrade/downgrade core packages until the error has been identified.

## 3. Run automated tests before real data

```bash
!ruff check .
!ruff format --check .
!pytest -q
```

The expected outcome is zero lint/format errors and all applicable tests passing. If pytest still fails to import `scripts.*`, capture the full error; the project import-path configuration is being remediated. Do not proceed to claim the test suite passes if collection or tests fail.

## 4. Locate and verify the approved CT-RATE input

Set these paths to the actual files visible in the Kaggle input pane. Do not guess their names:

```python
from pathlib import Path

for root in [Path("/kaggle/input"), Path("/kaggle/working")]:
    if root.exists():
        print(f"\n-- {root} --")
        for p in root.rglob("*"):
            if p.is_file() and ("metadata" in p.name.lower() or p.suffix.lower() == ".csv"):
                print(p)
```

Confirm the metadata CSVs have the required `VolumeName` column and that their referenced scans are resolvable by the streamer's supported source layout. Never paste private URLs or tokens into a report.

Example after replacing the paths with verified paths:

```python
import pandas as pd

train_meta = "/kaggle/input/APPROVED_INPUT_PATH/train_metadata.csv"
valid_meta = "/kaggle/input/APPROVED_INPUT_PATH/valid_metadata.csv"
for p in (train_meta, valid_meta):
    df = pd.read_csv(p)
    print(p, "rows=", len(df), "columns=", list(df.columns))
    assert "VolumeName" in df.columns, f"Missing VolumeName in {p}"
```

If scans are in Drive but not visible to Kaggle, stop here and switch to Colab or an approved transfer mechanism. Do not make restricted data publicly accessible just to make the smoke test work.

## 5. Run one-volume-per-split CT-RATE smoke extraction

First create a writable output directory with adequate free space:

```bash
!mkdir -p /kaggle/working/copd-smoke
```

Run the repository CLI, replacing metadata paths with verified paths:

```bash
!python scripts/stream_ctrate.py \
  --train-metadata "/kaggle/input/APPROVED_INPUT_PATH/train_metadata.csv" \
  --valid-metadata "/kaggle/input/APPROVED_INPUT_PATH/valid_metadata.csv" \
  --output-dir "/kaggle/working/copd-smoke/ctrate_smoke" \
  --tier lean \
  --max-volumes-per-split 1 \
  --patch-size 50 \
  --max-patches-per-scan 100
```

This should be a small real-data data-path check, not a research experiment. A successful process exit alone is insufficient.

## 6. Audit outputs and stop on any unexplained failure

```python
from pathlib import Path
import json

root = Path("/kaggle/working/copd-smoke/ctrate_smoke")
print("Exists:", root.exists())
for p in sorted(root.rglob("*")):
    if p.is_file():
        print(p.relative_to(root), p.stat().st_size, "bytes")
for name in ("stream_config.json", "stream_summary.json", "patch_counts.json"):
    p = root / name
    print("\n", name, "exists:", p.exists())
    if p.exists():
        print(p.read_text()[:12000])
```

Review `streaming.log`, `stream_log.csv`, every split's `patch_manifest.parquet`, selected volume/patient lists, and shard files. Check:
- each requested split has a recorded outcome;
- successful scans have nonzero, plausible patch counts;
- failed and no-patch scans are explained, not ignored;
- manifest shard paths exist and referenced indices are in range;
- patient IDs do not overlap across train/validation/test;
- the image intensity range is consistent with HU after the pipeline's conversion;
- lung masks are inside the scan, do not leak across exterior air, and have plausible volume;
- patch previews actually lie inside lung parenchyma.

The current threshold mask is provisional and is not a five-lobe segmentation. Do not create or report lobe-wise measurements from it.

## 7. Train only after the smoke audit passes

Run the preflight diagnostic against the **train split only**:

```bash
!python scripts/diagnose_training.py \
  --shards-dir "/kaggle/working/copd-smoke/ctrate_smoke/train" \
  --checkpoint-dir "/kaggle/working/copd-smoke/checkpoints" \
  --require-cuda \
  --report "/kaggle/working/copd-smoke/preflight_report.json"
```

Do not continue if its status is failed or if the patch manifest is empty. For the first GPU training smoke, use a separate real output directory and a short run (for example, 1 epoch, batch size 2) with the CLI's checkpoint options:

```bash
!python scripts/train_encoder.py \
  --shards-dir "/kaggle/working/copd-smoke/ctrate_smoke/train" \
  --checkpoint-dir "/kaggle/working/copd-smoke/checkpoints" \
  --epochs 1 --batch-size 2 --lr 0.0001 --temperature 0.07 \
  --device auto --checkpoint-every-steps 1 --require-cuda
```

Use this only if the manifest has enough valid patches for the contrastive loss and the diagnostic passes. A single epoch verifies the training path; it is not adequate model training or a performance result. Confirm that latest/best checkpoints and the training log are created, losses and gradients remain finite, and no exception was swallowed.

Only then fit a small density model from training embeddings, after checking the fit script's actual CLI and candidate-count requirements. Do not run real patient scoring yet: the repository currently explicitly marks real patient-level inference/evaluation as not implemented.

## 8. DIR-Lab smoke test is a separate validation

DIR-Lab data is paired inhale/exhale CT plus expert landmarks. Before running registration:
- confirm the ten cases' files are correctly paired and each landmark file has the expected count and coordinate convention;
- check NIfTI orientation, affine, spacing and intensity units;
- check inhale/exhale masks align with their own volumes;
- run a single case in fast mode first, then compute TRE from the correctly transformed landmarks in a shared physical coordinate system;
- report initial and final TRE in millimetres per case, and investigate any case with worsening TRE or implausible displacement.

Do not treat a registration output as valid merely because a warped NIfTI was written. PRM must use aligned images in consistent HU and geometry. Whole-lung masks are not lobe masks.

## 9. What is required for the full lobe-wise/exacerbation paper

CT-RATE and DIR-Lab alone do not supply the patient-level 12-month exacerbation outcomes needed to validate exacerbation prediction. To evaluate that claim, a separate approved clinical cohort must include patient identifiers that can be linked without leakage, baseline clinical covariates, follow-up duration, and timestamped exacerbation outcomes. Define the prediction index date, 365-day outcome window, censoring, inclusion/exclusion criteria, and missingness rules before modelling.

Until those labels exist, limit claims to validated imaging measurements and clearly labelled CT-RATE report-label proxy tasks. Never call report-label proxy AUC clinical COPD diagnostic accuracy or exacerbation prediction.

## 10. Preserve the evidence

Save a private run bundle containing:
- Git commit SHA and notebook/runtime package versions;
- selected scan and patient IDs per split;
- configuration and all manifests;
- mask/QC montages and scan-level success/failure log;
- preflight JSON, training logs, checkpoint metadata, and timing/disk observations;
- a short note listing failures and unresolved issues.

Do not commit raw scans, patient data, credentials, or model checkpoints to Git. The runbook is complete only when the results above are generated and inspected; these instructions alone are not evidence of a successful run.
