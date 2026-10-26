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

## Current code status (important)

- `scripts/ct_only_assessment.py` computes quantitative emphysema/density features by whole lung and lobe.
- `scripts/train_multilabel_head.py` trains a scan-level 18-label classifier from an already-produced embedding CSV and CT-RATE label CSV.
- The multi-label head **does not** generate embeddings from raw CT scans and **does not** predict per-lobe finding locations.
- A complete five-lobe multi-finding pipeline is therefore not implemented yet. This document is the target design; do not claim the desired end-to-end model exists until the raw-volume inference, spatial supervision, and tests are connected.

## Key literature

- Fleischner Society statement on CT-definable COPD subtypes: https://pmc.ncbi.nlm.nih.gov/articles/PMC4613878/
- Asian Society of Thoracic Radiology quantitative CT recommendations: https://pmc.ncbi.nlm.nih.gov/articles/PMC13389076/
- Review of quantitative CT and pulmonary function: https://publications.ersnet.org/content/errev/30/161/210031
