# COPD-Anomaly: Project Progress & Verification Tracker

## Current status (reviewed 2026-10-25)

**Research status: not yet ready for scientific evaluation.** The repository has modular components, but the prior milestone table overstated completion: several tests are synthetic unit tests, and some end-to-end paths were not connected to real CT-derived scores. Do not interpret prior "completed" labels as evidence of a validated end-to-end clinical pipeline.

| Area | Status | Evidence / next gate |
|---|---|---|
| Repository scaffold and dependency metadata | Implemented; CI not yet clean | Re-run Ruff and pytest on the remediation branch |
| DIR-Lab ingestion / coordinate geometry | Partially implemented; real runtime verification pending | Audit all 10 pairs, geometry, HU conversion and landmark conventions |
| Lung and five-lobe segmentation | Implementation added; real runtime/accuracy unverified | `scripts/segment_lobes.py` and CT-RATE `--segment-lobes` use named TotalSegmentator labels and fail closed; visually/QC-validate all five masks on real scans |
| Registration and TRE | Physical origin/direction now propagated; real DIR-Lab results unverified | Run all available landmarks; verify transform convention against known landmarks, inspect warped volumes and report per-case TRE |
| Patch extraction | Lobe-mask integration implemented; real output unverified | Optional `--lobe-masks-dir` in `make_patches.py` and `--segment-lobes` in CT-RATE streamer write actual lobe IDs; audit counts, HU values, split leakage, masks and coverage |
| Encoder training | Added resumable epoch checkpoints, history CSV, and 1-/2-channel input detection; no real training verified here | Train from real shards; inspect loss, checkpoint metadata, and stability over the intended run |
| Density fitting | Implementation present; depends on a real trained encoder and eligible real patches | Fit only on defined normal training patches; report counts and cohort selection |
| Real anomaly inference / proxy evaluation | Patch/scan/patient scoring and held-out proxy evaluator implemented; not run or validated on real data | Run inference on held-out real shards; provide a reviewed report-label table with a pre-specified extraction rule; audit outputs, patient joins, and held-out metrics |
| Lobe feature analysis | Real patch-score input path implemented; not validated on real data | Generate masks, extract patches with lobe IDs, score them, and run `scripts/lobe_analysis.py`; verify all five lobes are covered per scan |
| Synthetic multimodal / exacerbation ablations | **Out of scope for the imaging paper** | Do not spend publication-critical effort here; never present synthetic results as scientific evidence |
| Publication readiness | **Blocked** | Follow `docs/PUBLICATION_SCOPE.md`: clean CI, real-data pipeline, segmentation/geometry QC, patient-disjoint splits, held-out anomaly scores, whole-lung vs five-lobe comparisons, baselines and uncertainty reporting |

## Provenance rules

- Synthetic dry-run artefacts belong under `outputs/synthetic/` and must be labelled synthetic in their filenames or contents.
- Real outputs belong under `outputs/real/` and must be generated only from verified real inputs.
- A missing manifest, checkpoint, segmentation, or evaluation input is an error—not permission to generate random data silently.
- Synthetic AUROC, synthetic exacerbation outcomes, and synthetic lobe scores are not clinical evidence and must not appear in an abstract as results.
- DIR-Lab supports registration/landmark and imaging-method checks; it does not provide the clinical exacerbation outcome table needed to validate 12-month exacerbation prediction.
- CT-RATE report labels are proxy labels, not confirmed COPD diagnosis or clinical outcomes.

## Immediate execution order

1. Run CI on this branch and fix actual test/lint failures. Latest CI status must be checked after the final code commit; do not treat CI as passing until both Python jobs complete successfully.
2. Audit DIR-Lab NIfTI geometry/HU conversion; validate the TotalSegmentator label map and audit registration/TRE on real landmarks.
3. Generate and QC real patch shards; ensure patient-level split integrity and document selection.
4. Train encoder on real patches; save reproducible config and loss history. Training may take 1–2 days depending on GPU, data volume and patch count.
5. Fit density model only from eligible training-set embeddings.
6. Run real patch scoring and patient-level aggregation on held-out real scans.
7. Run five-lobe segmentation and lobe-aware patch extraction, then verify missing-lobe handling and per-lobe coverage.
8. Compare whole-lung scores, five-lobe summaries, and a pre-specified imaging baseline on held-out patients; add uncertainty and robustness checks.
9. Prepare reproducible figures, tables, configuration, and limitations. Clinical/exacerbation modeling is outside this paper’s scope, not a release blocker.

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
- No real CT volumes, masks, landmarks, or patch shards have yet been processed in this environment. New scoring/segmentation code remains unverified until smoke-tested.
- No real-data metrics or clinical performance claims have been established.

Update this file only when a result is reproducibly verified; include command, dataset split, relevant counts, and CI run/commit.


## Additional implementation changes (2026-10-25)

- Added strict validation for five-lobe masks and an explicit TotalSegmentator-based lobe-generation CLI. TotalSegmentator label IDs are mapped by documented anatomical class names; unexpected IDs fail instead of being guessed.
- Added optional lobe-aware CT-RATE streaming via `--segment-lobes` and optional DIR-Lab patch extraction via `--lobe-masks-dir`. The default threshold-only path remains global-only and must not be used for lobe-wise claims.
- Added real patch anomaly scoring with scan/patient aggregation, optional GMM/flow scoring, and held-out report-label proxy evaluation. Scores are marked `REAL_CT_PATCH_INFERENCE_UNVALIDATED`; metrics are labelled report-label proxy, not clinical diagnosis.
- Updated registration to preserve NIfTI origin/direction and reject implicit shape-mismatch resampling, which would invalidate landmark comparisons.
- Added `docs/CLOUD_RUNTIME.md` describing path-based Colab/Kaggle execution and `docs/KAGGLE_REAL_DATA_SMOKE.md` for Kaggle smoke validation.
- Publication scope is narrowed to imaging-based anomaly quantification and five-lobe analysis; see `docs/PUBLICATION_SCOPE.md`. Exacerbation prediction, clinical outcome modeling, and multimodal fusion are out of scope for this paper.
- A CT-RATE report-label proxy analysis is optional and secondary only; it requires a reviewed, reproducible label-extraction table and a held-out real-data run, and cannot be described as clinical diagnosis.


## Five-lobe multi-finding milestone (2026-10-26)

- Added `scripts/build_five_lobe_features.py`: extracts per-lobe HU distribution, low-attenuation percentages, and volume from CT NIfTI plus matching five-lobe masks; emits a long-format feature table for modelling.
- Added `scripts/train_five_lobe_mil.py`: trains an exploratory 18-label noisy-OR MIL model from per-lobe features and scan-level CT-RATE labels; emits patient-grouped holdout metrics and per-lobe weak-localization hypotheses.
- Added synthetic tests for the feature extractor and MIL training. CI was observed queued/in progress after the implementation commits; do not mark it passing until all jobs finish.
- Important limit: no raw-volume end-to-end encoder is connected to this MIL head, and no radiologist-annotated lobe-level findings are available in the current workflow. The lobe scores are hypotheses, not verified localization or clinical severity.
- Dataset decision: DIR-Lab COPDgene's 10 paired scans are not the primary training data because they lack CT-RATE's 18 finding labels. Keep them only as optional registration/TRE and exploratory paired-inspiratory/expiratory PRM validation; do not mix them into CT-RATE training or evaluation.
