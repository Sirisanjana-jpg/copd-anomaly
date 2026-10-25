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

### Optional lobe-wise CT-RATE extraction

The default threshold path intentionally creates **global-only** patches (lobe ID `-1`). To create lobe-wise patch metadata, install the optional segmentation dependency and use a fresh output directory:

```bash
python -m pip install -e ".[dev,segmentation]"
python scripts/stream_ctrate.py --train-metadata data/raw/ctrate/train_metadata.csv --valid-metadata data/raw/ctrate/valid_metadata.csv --output-dir data/processed/ctrate_lobes_smoke --tier lean --max-volumes-per-split 1 --patch-size 50 --max-patches-per-scan 100 --segment-lobes
```

TotalSegmentator is run on each selected scan and its named lung-lobe classes are mapped to project labels: 1=RUL, 2=RML, 3=RLL, 4=LUL, 5=LLL. The code fails closed if it cannot produce a label map; it does not substitute a threshold mask. This is slower than threshold segmentation. Review each label mask and patch coverage before any training. A valid label map is still not a guarantee of anatomical accuracy; segmentation quality must be manually/quantitatively checked before publication. Do not resume a non-lobe run with `--segment-lobes` or vice versa; the run config is intentionally different.

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

If using CPU intentionally, omit `--require-cuda` from both the diagnostic and training commands. Do not proceed if the report status is `failed`; inspect the listed errors first. The training CLI also checks CUDA so it will not silently fall back to a slow CPU run.

## 4. Train the encoder

Train only on the `train` shards, not the validation/test shards. Start with a short run to catch memory/shape issues; then choose the intended epoch count before a long run. A 3D encoder run may take many hours to days depending on GPU, patch count and I/O.

```powershell
python scripts/train_encoder.py --shards-dir data/processed/ctrate_lean/train --checkpoint-dir outputs/real/checkpoints --epochs 30 --batch-size 4 --lr 0.0001 --temperature 0.07 --device auto --checkpoint-every-steps 25 --require-cuda
```

The script detects whether patches have one input channel or paired channels and writes:
- `outputs/real/checkpoints/encoder_best.pt` — best epoch by training contrastive loss
- `outputs/real/checkpoints/encoder_latest.pt` — atomic recovery checkpoint, including optimizer/scheduler and RNG states; saved every 25 batches and at each completed epoch
- `outputs/real/checkpoints/training_history.csv` — per-epoch loss, batch count, and elapsed seconds
- `outputs/real/checkpoints/training.log` — persistent console output, failure traceback, and batch/scan context

If interrupted, resume using the **same total epoch count and training settings**. Recovery resumes from the latest saved batch checkpoint when available, so a crash should lose at most the most recent checkpoint interval of work. Keep `--checkpoint-every-steps` the same for consistent operational behavior:

```powershell
python scripts/train_encoder.py --shards-dir data/processed/ctrate_lean/train --checkpoint-dir outputs/real/checkpoints --epochs 30 --batch-size 4 --lr 0.0001 --temperature 0.07 --device auto --checkpoint-every-steps 25 --require-cuda --resume-from outputs/real/checkpoints/encoder_latest.pt
```

Use `--device cuda` only when CUDA is available. Reduce batch size if GPU memory is insufficient; keep batch size at least 2 for NT-Xent. Archive the code commit, selected patient IDs, shard manifests, environment/package versions, GPU model, seed and training history.

## 5. Fit the density model

Fit the density estimator only from eligible patches in the training split:

```powershell
python scripts/fit_density.py --shards-dir data/processed/ctrate_lean/train --encoder-ckpt outputs/real/checkpoints/encoder_best.pt --output-model outputs/real/checkpoints/density_model.joblib --model-type gmm --components 8
```

Check `outputs/real/checkpoints/density_fit.log` and how many patches pass the LAA-950 filter before interpreting the result. If too few eligible patches remain, do not silently loosen the threshold; document a scientifically justified selection rule and rerun with a fresh, versioned output.

## 6. Real inference and held-out report-label proxy evaluation

After a real encoder and GMM density model have been trained/fitted, score the held-out test shards (do not score training shards for the final reported metrics):

```bash
python scripts/score_anomaly.py --real-inference --shards-dir data/processed/ctrate_lean/test --encoder-ckpt outputs/real/checkpoints/encoder_best.pt --density-model outputs/real/checkpoints/density_model.joblib --output-dir outputs/real/reports --device auto
```

This writes patch-, scan-, and patient-level scores with the split and provenance marked. It does not assign report labels, diagnose COPD, or predict exacerbations. The density model must have real-training provenance; legacy/synthetic density artifacts are rejected.

To evaluate a CT-RATE report-label proxy, first create and review a separate label table from the approved report metadata using a documented, versioned rule. Required columns are `patient_id,label` and optionally `split`; label must be binary 0/1. Do not use the test scores to choose the label rule, score threshold, or model. The label file must refer to exactly the patients in the held-out score file.

```bash
python scripts/evaluate_proxy.py --patient-scores outputs/real/reports/patient_anomaly_scores.csv --labels data/processed/ctrate_test_report_labels.csv --output outputs/real/reports/report_label_proxy_metrics.json --label-definition "REPLACE_WITH_REVIEWED_RULE_AND_VERSION" --score-column anomaly_mean --expected-split test
```

The evaluator rejects training/validation scores, duplicate patients, nonbinary labels, mismatched patient IDs, and unverified score provenance. Output is explicitly labelled `CT_RATE_REPORT_LABEL_PROXY_NOT_CLINICAL_DIAGNOSIS`. It is a report-derived proxy benchmark only.

CT-RATE report labels are not confirmed COPD diagnoses or clinical outcomes. DIR-Lab can support registration/landmark and imaging-method checks, but it does not provide the clinical 12-month exacerbation outcomes required to validate an exacerbation-risk model. Clinical COPD classification or exacerbation prediction requires an appropriate real clinical cohort and outcomes.

Do not upload gated CT data or model artifacts to a public repository without checking dataset terms.
