# Real-data training runbook

This runbook describes how to prepare a real CT-RATE patch dataset and train the encoder. It is a procedure, not evidence that training has already run. GitHub CI uses small synthetic unit tests; do not train a research model inside CI.

## 0. Verify access and environment

Use a machine with sufficient disk/RAM and preferably a CUDA-capable GPU. Activate Python 3.10 or 3.11 and install the project:

```powershell
python -m pip install -e ".[dev]"
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

You must have approved access to the gated CT-RATE dataset and its metadata CSVs. Set the token in your shell without placing it in source code or committing it:

```powershell
$env:HF_TOKEN = Read-Host "Enter your Hugging Face read token"
```

Use the actual metadata CSV locations on your machine for the `--train-metadata` and `--valid-metadata` arguments. The expected column is `VolumeName`. The script creates patient-disjoint train/validation/test selections, writes the selected volume lists and split IDs, logs each volume, deletes temporary downloaded NIfTI files, and can resume from saved patch manifests.

## 1. Run a small CT-RATE smoke pass first

Start with a one-volume-per-split test to verify gated access, remote paths, intensity scaling, resampling, segmentation and shard output before starting a long run:

```powershell
python scripts/stream_ctrate.py --train-metadata data/raw/ctrate/train_metadata.csv --valid-metadata data/raw/ctrate/valid_metadata.csv --output-dir data/processed/ctrate_smoke --tier lean --max-volumes-per-split 1 --patch-size 50 --max-patches-per-scan 100
```

Replace the two metadata paths with your real file locations. The smoke pass uses real CT-RATE volumes, not random data, but it is **not** a scientifically sufficient training cohort.

The streamer canonicalizes image orientation, resamples intensities to a 1 mm isotropic grid, checks for non-finite and implausible non-HU intensity ranges, derives a threshold lung mask, extracts patches and writes per-split manifests. It logs failed volumes instead of silently counting them as successful. Inspect the masks and patch locations visually before scaling up. The current threshold mask is a baseline, not validated five-lobe segmentation; do not report lobe-wise findings from it.

Outputs are stored in separate directories:
- `data/processed/ctrate_smoke/train/`
- `data/processed/ctrate_smoke/val/`
- `data/processed/ctrate_smoke/test/`

Each split contains `patch_manifest.parquet`, shard files, `selected_volumes.csv`, `selected_patient_ids.txt`, and `stream_log.csv`. The output root also contains `streaming.log` with stage-specific stack traces. Review `patch_counts.json`, `stream_config.json`, `stream_summary.json`, and all `failed` / `no_patches` log rows. Failed rows include the exception type, stage, message, and traceback; the summary lists per-split status counts and failed volume names. The current code filters training density candidates by LAA-950 only; this is a heuristic, not the full CT-RATE report-negative normality definition described in the project plan.

## 2. Stream the intended cohort

After the smoke pass and mask QC, run the intended tier. The configured `lean` tier selects 100 train, 30 validation and 70 test subjects; check that the source metadata actually contains enough unique subjects. Use a fresh output directory for a new selection, or rerun against the same directory to resume:

```powershell
python scripts/stream_ctrate.py --train-metadata data/raw/ctrate/train_metadata.csv --valid-metadata data/raw/ctrate/valid_metadata.csv --output-dir data/processed/ctrate_lean --tier lean --patch-size 50 --max-patches-per-scan 100 --shard-size 1000
```

The stream CLI saves manifests after each successfully processed scan and records failures for review. It does not make an inaccessible dataset accessible; if gated access or remote file paths fail, fix those first. The current streamer uses threshold segmentation and must be considered provisional until masks are reviewed. Keep the split directories separate throughout training and evaluation.

**Do not use `scripts/make_patches.py` for CT-RATE pretraining.** That script currently scans DIR-Lab-style filenames matching `*_insp.nii.gz`; it is intended for DIR-Lab paired-scan work and QC, not for the CT-RATE cohort.

## 3. Preflight the shards before training

Run the diagnostic script after the complete cohort has been streamed and reviewed. It checks package availability, free disk space, manifest/shard integrity, referenced indices, sample patch shapes, CUDA access if requested, and checkpoint-directory writability. It does not start training.

```powershell
python scripts/diagnose_training.py --shards-dir data/processed/ctrate_lean/train --checkpoint-dir outputs/real/checkpoints --require-cuda --report outputs/real/preflight_report.json
```

If using CPU intentionally, omit `--require-cuda`. Do not proceed if the report status is `failed`; inspect the listed errors first.

## 4. Train the encoder

Train only on the `train` shards, not the validation/test shards. Start with a short run to catch memory/shape issues; then choose the intended epoch count before a long run. A 3D encoder run may take many hours to days depending on GPU, patch count and I/O.

```powershell
python scripts/train_encoder.py --shards-dir data/processed/ctrate_lean/train --checkpoint-dir outputs/real/checkpoints --epochs 30 --batch-size 4 --lr 0.0001 --temperature 0.07 --device auto
```

The script detects whether patches have one input channel or paired channels and writes:
- `outputs/real/checkpoints/encoder_best.pt` — best epoch by training contrastive loss
- `outputs/real/checkpoints/encoder_latest.pt` — latest epoch, optimizer/scheduler state, and history for resuming
- `outputs/real/checkpoints/training_history.csv` — per-epoch loss, batch count, and elapsed seconds
- `outputs/real/checkpoints/training.log` — persistent console output, failure traceback, and batch/scan context

If interrupted, resume using the **same total epoch count and settings**:

```powershell
python scripts/train_encoder.py --shards-dir data/processed/ctrate_lean/train --checkpoint-dir outputs/real/checkpoints --epochs 30 --batch-size 4 --lr 0.0001 --temperature 0.07 --device auto --resume-from outputs/real/checkpoints/encoder_latest.pt
```

Use `--device cuda` only when CUDA is available. Reduce batch size if GPU memory is insufficient; keep batch size at least 2 for NT-Xent. Archive the code commit, selected patient IDs, shard manifests, environment/package versions, GPU model, seed and training history.

## 5. Fit the density model

Fit the density estimator only from eligible patches in the training split:

```powershell
python scripts/fit_density.py --shards-dir data/processed/ctrate_lean/train --encoder-ckpt outputs/real/checkpoints/encoder_best.pt --output-model outputs/real/checkpoints/density_model.joblib --model-type gmm --components 8
```

Check `outputs/real/checkpoints/density_fit.log` and how many patches pass the LAA-950 filter before interpreting the result. If too few eligible patches remain, do not silently loosen the threshold; document a scientifically justified selection rule and rerun with a fresh, versioned output.

## 6. Current pipeline boundary

Real patient-level scoring and held-out proxy evaluation are **not implemented end-to-end yet**. The scoring script intentionally fails rather than writing an empty or fabricated real report. Successful encoder training and density fitting alone are not evidence of diagnostic performance. Next work is to implement real patch inference, patient-level aggregation, held-out report-label proxy evaluation, and validated lobe mapping.

CT-RATE report labels are proxy labels, not confirmed COPD diagnoses or clinical outcomes. DIR-Lab can support registration/landmark and imaging-method checks, but it does not provide the clinical 12-month exacerbation outcomes required to validate an exacerbation-risk model. Clinical COPD classification or exacerbation prediction requires a real clinical cohort with appropriate labels/outcomes.

Do not upload gated CT data or model artifacts to a public repository without checking dataset terms.
