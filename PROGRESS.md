# COPD-Anomaly: Project Progress & Verification Tracker

## Current status (reviewed 2026-10-25)

**Research status: not yet ready for scientific evaluation.** The repository has modular components, but the prior milestone table overstated completion: several tests are synthetic unit tests, and some end-to-end paths were not connected to real CT-derived scores. Do not interpret prior "completed" labels as evidence of a validated end-to-end clinical pipeline.

| Area | Status | Evidence / next gate |
|---|---|---|
| Repository scaffold and dependency metadata | Implemented; CI not yet clean | Re-run Ruff and pytest on the remediation branch |
| DIR-Lab ingestion / coordinate geometry | Partially implemented; real runtime verification pending | Audit all 10 pairs, geometry, HU conversion and landmark conventions |
| Lung and five-lobe segmentation | **Unverified / not publication-ready** | Validate backend output and all five labels; reject binary-only masks for lobe analysis |
| Registration and TRE | Transform direction corrected in remediation branch; real DIR-Lab results unverified | Confirm coordinate units, run all available landmarks, inspect warped volumes and report per-case TRE |
| Patch extraction | Added fail-fast input/geometry/empty-output checks in remediation branch; lobe assignment still not connected | Run extraction on real data; check counts, HU values, patient-level splits, masks and lobe IDs |
| Encoder training | Added resumable epoch checkpoints, history CSV, and 1-/2-channel input detection; no real training verified here | Train from real shards; inspect loss, checkpoint metadata, and stability over the intended run |
| Density fitting | Implementation present; depends on a real trained encoder and eligible real patches | Fit only on defined normal training patches; report counts and cohort selection |
| Real anomaly inference / proxy evaluation | **Not implemented end-to-end**; now fails closed instead of returning an empty report | Connect real patches → encoder embeddings → density scores → patient aggregation → held-out report labels |
| Lobe feature analysis | Previously fabricated lobe IDs/scores; remediation branch blocks real output until inputs are connected and preserves missing lobes as missing | Connect validated lobe masks and real patch scores before generating real feature tables |
| Synthetic multimodal / exacerbation ablations | Synthetic software tests only | Never present as clinical evidence; requires a real labelled cohort for clinical claims |
| Publication readiness | **Blocked** | Clean CI, reproducible real-data pipeline, cohort-level leakage checks, validated registration/segmentation and held-out evaluation |

## Provenance rules

- Synthetic dry-run artefacts belong under `outputs/synthetic/` and must be labelled synthetic in their filenames or contents.
- Real outputs belong under `outputs/real/` and must be generated only from verified real inputs.
- A missing manifest, checkpoint, segmentation, or evaluation input is an error—not permission to generate random data silently.
- Synthetic AUROC, synthetic exacerbation outcomes, and synthetic lobe scores are not clinical evidence and must not appear in an abstract as results.
- DIR-Lab supports registration/landmark and imaging-method checks; it does not provide the clinical exacerbation outcome table needed to validate 12-month exacerbation prediction.
- CT-RATE report labels are proxy labels, not confirmed COPD diagnosis or clinical outcomes.

## Immediate execution order

1. Run CI on this branch and fix actual test/lint failures. A one-time formatter pass is being attempted; do not treat CI as passing until the final run completes.
2. Audit DIR-Lab NIfTI geometry, HU conversion, segmentation and registration direction.
3. Generate and QC real patch shards; ensure patient-level split integrity and document selection.
4. Train encoder on real patches; save reproducible config and loss history. Training may take 1–2 days depending on GPU, data volume and patch count.
5. Fit density model only from eligible training-set embeddings.
6. Implement real patch scoring and patient-level aggregation; evaluate report-label proxy only on a held-out set and label it correctly.
7. Connect validated lobe labels to patch scores; test missing-lobe handling and per-lobe coverage.
8. Keep clinical/exacerbation modeling blocked until a real cohort with outcome labels is available.

## Verification log

### Verified from repository inspection
- Source files and tests are present in the GitHub repository.
- The earlier CI run failed Ruff (58 findings); pytest did not run in that job. This is not a passing CI result.
- The previous `score_anomaly.py` real-model branch returned an empty DataFrame rather than performing real patient scoring.
- The previous `lobe_analysis.py` generated random scores and fabricated lobe IDs regardless of the supplied manifest.
- The previous encoder and density scripts silently switched to synthetic data when manifests were missing.
- `make_patches.py` passes `lobe_mask=None`, so real lobe assignment is not connected in that path.

### Not verified in this review
- No local execution of the full test suite was available through the GitHub connector.
- No training was run here, and no new encoder/density checkpoint was validated.
- No real CT volumes, masks, landmarks, or patch shards were processed during this review.
- No real-data metrics or clinical performance claims have been established.

Update this file only when a result is reproducibly verified; include command, dataset split, relevant counts, and CI run/commit.
