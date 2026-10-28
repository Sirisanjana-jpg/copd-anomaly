# Publication scope: CT anomaly detection and five-lobe analysis

## In-scope research question

Can self-supervised CT patch representations and density-based anomaly scores provide reproducible quantitative characterization of COPD-related CT patterns, and does aggregating those scores by the five anatomical lung lobes add useful spatial interpretability beyond whole-lung summaries?

This paper is an **imaging-method and anomaly-quantification study**. It does not claim to predict exacerbations, diagnose an individual, or establish clinical utility.

## Included

1. Real chest CT preprocessing and quality control.
2. Validated lung and five-lobe segmentation.
3. Patch extraction with patient-disjoint train/validation/test splits.
4. Encoder training on real training patches, with saved configuration, seed, checkpoint provenance, and loss history.
5. Density fitting on training-set embeddings only, followed by anomaly scoring on held-out scans.
6. Whole-lung, scan-level, and five-lobe summaries from the same patch scores.
7. Quantitative comparison of global versus lobe-wise analysis, reproducibility checks, uncertainty estimates, and representative visual quality-control figures.

## Explicitly out of scope for this paper

- Exacerbation-risk or future-outcome prediction.
- Claims about clinical diagnosis or treatment decisions.
- Synthetic-cohort results as scientific evidence.
- Multimodal or clinical fusion experiments.
- Any unverified report-label proxy as a primary clinical endpoint.

A report-derived label analysis may be included only as a clearly labelled exploratory secondary analysis if the label table has a documented, reviewed extraction rule and is evaluated on a held-out patient-disjoint test split. Such labels remain report-label proxies, not ground-truth diagnosis.

## Minimum experimental protocol

### A. Data and leakage controls

- Document the dataset release/version, inclusion criteria, scan counts, patient counts, and any exclusions.
- Split at the **patient level** before fitting any learned component; keep all scans from a patient in one split.
- Fit preprocessing statistics, encoder, and density model using training data only. Use validation data for model/threshold choices and keep the test split untouched until final evaluation.
- Record random seeds, dependency versions, configuration, hardware, checkpoint hashes, and exact commands.
- Report failed scans and exclusions rather than silently dropping them.

### B. Segmentation and anatomical QC

- Generate five-lobe masks with the named TotalSegmentator lobe classes; do not infer lobes from a threshold-only lung mask.
- Review overlays on representative scans and report per-lobe coverage, empty/invalid masks, and failure rates.
- Check image/mask affine, orientation, spacing, and resampling alignment.
- Do not describe segmentation as validated solely because a mask contains labels 1–5. If no independent reference masks are available, state that validation is QC-based and do not claim quantitative segmentation accuracy.

### C. Anomaly method and comparators

Use the same held-out scans and preprocessing for each comparison:

1. **Simple imaging baseline:** LAA-950 fraction or another pre-specified, transparent CT attenuation summary.
2. **Whole-lung anomaly model:** aggregate all eligible lung patches per scan.
3. **Five-lobe anomaly analysis:** report each lobe separately and provide pre-specified whole-lung aggregation for a fair comparison.
4. **Density-model comparison:** compare GMM and RealNVP only if both are trained under the same split and protocol.

Define patch eligibility, lung fraction filters, normal-patch selection, score aggregation, and all hyperparameters before test evaluation. LAA-950 selection is a heuristic and must not be called verified healthy-control selection.

### D. Results to report

- Cohort and split counts, scan failures, patch counts, and lobe coverage.
- Training curves and model/checkpoint provenance.
- Distribution and stability of patch-, scan-, and lobe-level anomaly scores.
- Association of anomaly scores with available, independently defined imaging or report-derived reference variables, if valid references exist.
- Confidence intervals and patient-level resampling where inferential metrics are appropriate.
- Robustness to random seed, patch sampling, missing/low-coverage lobes, and reasonable preprocessing choices.
- Representative CT/lobe-mask overlays and anomaly-score visualizations with identifiers removed.

If there are no valid independent labels or annotations, present the work as unsupervised quantitative characterization and do not invent AUROC, localization accuracy, or disease-severity ground truth. Visual heatmaps alone do not prove lesion-localization accuracy.

## Publication gates

The paper is not ready until all of the following are evidenced:

- CI passes on the publication branch.
- Real-data smoke run completes and all failures are reviewed.
- Segmentation and geometry QC pass on a documented sample.
- Patient-level split leakage checks pass.
- Encoder and density model are trained on real training data and their provenance is verified.
- Held-out anomaly scores and lobe summaries are reproducibly generated.
- Baselines and ablations are completed with appropriate uncertainty reporting.
- Figures, tables, limitations, and a rerunnable experiment record are prepared.

A successful unit-test suite is software evidence, not scientific validation. Keep all unvalidated results labelled as such in logs and reports.
