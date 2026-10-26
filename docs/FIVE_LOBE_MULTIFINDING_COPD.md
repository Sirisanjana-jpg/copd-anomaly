# Five-lobe, multi-finding COPD CT phenotyping: research plan

## Primary objective

Build a CT-based research system that estimates **which COPD-related structural abnormalities are present, where they are distributed across the five lung lobes, and how much CT-visible abnormality burden is present per lobe and per patient**. Do not reduce the project to emphysema-only analysis.

This is a multi-label, spatially resolved phenotype task. A patient may have several findings at once and different lobes may have different burdens. The intended output is a five-lobe table and a scan-level summary, not one mutually exclusive disease class.

## Target findings

Prioritize CT findings with a plausible COPD structural phenotype relationship and which are represented in CT-RATE labels:

- Emphysema
- Mosaic attenuation pattern (possible small-airway/air-trapping phenotype; expiratory CT strengthens interpretation)
- Peribronchial thickening (airway-wall phenotype)
- Bronchiectasis
- Atelectasis
- Lung opacity and consolidation (non-specific findings; report separately, not as COPD-specific evidence)
- Pulmonary fibrotic sequela and interlobular septal thickening (important differential/coexisting abnormalities, not COPD-specific)
- Lung nodules and pleural effusion can be tracked as additional findings, but should not be folded into a COPD severity score by default.

The remaining CT-RATE categories (cardiomegaly, vascular/coronary calcification, hiatal hernia, lymphadenopathy, pericardial effusion, medical material) can be retained in a general chest-finding head, but they are not COPD severity components.

## What the data can supervise

CT-RATE's 18 labels are scan/report-level labels. They do **not** directly tell us which of the five lobes contains a finding, provide voxel-level lesion masks, or provide a calibrated COPD severity grade. Therefore:

1. Train/evaluate scan-level multi-label prediction against CT-RATE report-derived labels as weak-label baseline.
2. Segment the five lobes and extract per-lobe imaging features and local patch embeddings.
3. For spatial prediction, use only defensible supervision: lobe-annotated data, radiologist-reviewed localization, or a carefully designed weakly supervised multiple-instance learning approach. If scan-level labels are weakly assigned to individual lobes, that is label leakage/noisy supervision and must not be presented as verified lobe localization.
4. Evaluate localization separately from scan-level classification. Classification AUROC alone does not validate the predicted lobe.

## Per-lobe feature families

### Parenchymal density / emphysema
- LAA-950 percentage, LAA-910 percentage, mean/median HU, 15th percentile (Perc15), standard deviation, lobe volume.
- Keep thresholds and scan acquisition/reconstruction settings explicit. These measures are quantitative imaging biomarkers, not a universal clinical severity scale.

### Small-airway / air-trapping phenotype
- Inspiratory CT: texture/attenuation features and mosaic-attenuation prediction may be explored, with uncertainty.
- Paired registered expiratory CT: expiratory low attenuation (commonly LAA-856) and PRM-style analysis can better characterize air trapping / functional small-airway disease.
- CT-RATE does not guarantee paired expiratory acquisitions. Do not claim PRM or direct expiratory air trapping where only inspiratory CT exists.

### Airway phenotype
- Peribronchial thickening / airway-wall features and bronchiectasis prediction.
- Robust airway morphometry requires airway segmentation and sufficient spatial resolution; lobe masks alone cannot provide airway-wall measurements.

### Other abnormalities
- Atelectasis, opacity, consolidation, fibrosis and nodules require spatial models/annotations for trustworthy lobe-level localization. Scan-level report labels alone only support scan-level weak-label evaluation.

## Patient-level severity: define the endpoint carefully

Create an interpretable **CT structural burden profile**, not an invented GOLD grade. It can summarize per-lobe quantitative features and calibrated probabilities for each target finding, with uncertainty and a transparent aggregation rule. Do not combine unrelated findings into one score without prespecified clinical justification and validation.

If the later goal is to predict clinical COPD severity, that requires patient-level reference outcomes such as post-bronchodilator spirometry (FEV1, FEV1/FVC), symptoms, or validated clinical outcomes. CT-derived burden may be a predictor, but it is not itself a clinically validated measure of airflow limitation. GOLD grade is deliberately out of scope for this phase.

## Recommended implementation order

1. **Data audit:** inspect CT-RATE label CSV columns, scan IDs, patient IDs, class prevalence, and split definitions; ensure patient-disjoint splits and match embeddings to the correct volume.
2. **Lobe segmentation and QC:** validate the five masks on each CT and record mask quality/coverage. DIR-Lab's 10 cases are not enough to train or clinically validate a multi-disease model.
3. **Baseline scan-level multi-label model:** train from image-derived features/embeddings, report per-label AUROC, average precision, calibration, and confidence intervals against report-derived labels.
4. **Five-lobe quantitative feature table:** extract lobe features for all usable scans, retaining scan acquisition metadata and quality-control flags.
5. **Spatial model:** add lobe-level supervision or multiple-instance learning, then evaluate localization using lobe-annotated/reference data. Keep output marked as weakly localized until this succeeds.
6. **CT burden profile:** create transparent per-finding and per-lobe burden summaries; do not present as clinical severity without external validation.
7. **External validation:** when a cohort with suitable annotations and spirometry becomes available, evaluate on held-out patients and assess association with clinical measures.

## Implemented baseline and current limits

- `scripts/ct_only_assessment.py` computes quantitative density/emphysema features by whole lung and lobe.
- `scripts/train_multilabel_head.py` trains a scan-level 18-label classifier from precomputed scan embeddings and CT-RATE report-derived labels.
- `scripts/build_five_lobe_features.py` extracts quantitative CT features per lobe from a NIfTI CT and a matching five-lobe mask, yielding the long-format input table for spatial modelling.
- `scripts/train_five_lobe_mil.py` adds a **weakly supervised five-lobe multiple-instance learning (MIL) baseline**. It takes numeric image-derived features for each of the five lobes, predicts all 18 findings at lobe level, and combines the five probabilities with noisy-OR to train against scan-level labels. It writes scan-level metrics, a model checkpoint, and `holdout_lobe_hypotheses.csv`.
- The lobe scores are **hypotheses**, not validated localization: only scan-level report labels supervise training. No lobe-annotated reference set is currently wired into evaluation.
- The current model consumes precomputed lobe features; it does **not** read raw CT, segment lobes, extract learned 3D embeddings, or provide voxel-level lesion masks. Those remain integration milestones.
- Clinical COPD severity is not predicted or validated by this baseline. It reports CT finding predictions only; spirometry/clinical outcomes are needed to validate a clinical severity endpoint.

### Running the MIL baseline

First generate the numeric lobe feature table from the actual CT volumes and their corresponding masks. The CT and mask filenames must match; masks must be named `<VolumeName>_lobes.nii.gz`. CT/mask geometry and lobe IDs 1–5 are validated. The resulting table has one row per lobe with `VolumeName`, `patient_id`, `lobe_id` (`RUL`, `RML`, `RLL`, `LUL`, `LLL`) and numeric HU/volume features. The labels CSV must contain the 18 CT-RATE label columns and a matching volume-name column.

```powershell
python scripts/build_five_lobe_features.py --ct-dir data/ctrate_nifti --lobe-dir data/ctrate_lobes --output-csv outputs/real/reports/ct_lobe_feature_table.csv
python scripts/train_five_lobe_mil.py `
  --lobe-features-csv outputs/real/reports/ct_lobe_feature_table.csv `
  --labels-csv dataset/multi_abnormality_labels/train_predicted_labels.csv `
  --output-dir outputs/real/five_lobe_mil `
  --epochs 100
```

Outputs:
- `five_lobe_mil.pt`: model weights and feature normalization metadata.
- `metrics.json`: patient-grouped holdout scan-level metrics and explicit validation caveats.
- `holdout_lobe_hypotheses.csv`: one row per held-out scan/lobe with the raw input features and 18 weak localization scores.
- `holdout_scan_burden_profile.csv`: a scan-level summary of volume-weighted density features and scan-level probabilities. This is a transparent CT structural profile, not a clinical severity score.

Medical-imaging MIL is a reasonable research strategy when only case-level labels exist, but global labels do not prove the model has found the actual abnormal lobe. Independent lobe-level annotation is required to quantify localization performance.

## Key literature

- Fleischner Society statement on CT-definable COPD subtypes: https://pmc.ncbi.nlm.nih.gov/articles/PMC4613878/
- Asian Society of Thoracic Radiology quantitative CT recommendations: https://pmc.ncbi.nlm.nih.gov/articles/PMC13389076/
- Review of quantitative CT and pulmonary function: https://publications.ersnet.org/content/errev/30/161/210031


## Dataset decision: DIR-Lab COPDgene

DIR-Lab COPDgene's 10 paired inspiratory/expiratory cases are **not part of the primary multi-label training cohort**. They do not supply the 18 CT-RATE finding labels, and ten patients are far too few to train or validate a multi-finding model. Their strongest value is as a small, optional technical benchmark for inhale/exhale registration using expert landmarks and, if the paired scans and registration are sound, exploratory air-trapping/PRM feature development. We therefore keep the code/data path only as an optional registration/PRM validation asset; do not spend the primary modelling budget on it or mix its cases into CT-RATE training/evaluation. See [DIR-Lab's dataset description](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/copdgene.html) and the registration/air-trapping study ([Radiology: Artificial Intelligence](https://pmc.ncbi.nlm.nih.gov/articles/PMC8823452/)).
