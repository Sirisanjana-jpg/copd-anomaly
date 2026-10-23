# Real Clinical Cohort Specification & Data Dictionary

## 1. Prospective Cohort Protocol Requirements

To deploy the **COPD-Anomaly** pipeline in real clinical trials or prospective observational studies (such as COPDGene, SPIROMICS, or ECLIPSE), data ingestion must adhere to the standardized schema defined below.

### 1.1 Inclusion & Exclusion Criteria
- **Inclusion**:
  - Age $\ge 40$ years old.
  - History of cigarette smoking $\ge 10$ pack-years.
  - Baseline pre- and post-bronchodilator spirometry (FEV1, FVC).
  - Paired inspiratory and expiratory volumetric thin-slice non-contrast chest CT.
- **Exclusion**:
  - Active pulmonary tuberculosis, untreated lung malignancy, or severe metal artifacts obscuring thoracic anatomy.
  - Inability to perform acceptable and repeatable spirometry per ATS/ERS 2019 guidelines.

---

## 2. Standardized Clinical Feature Schema

| Field Name | Type | Unit / Format | Permissible Range | Clinical Definition & Standard |
|---|---|---|---|---|
| `patient_id` | String | Identifier | Unique string | De-identified subject identifier |
| `age` | Float | Years | 40.0 – 95.0 | Age at baseline examination |
| `sex` | Categorical | 'M', 'F' | {'M', 'F'} | Biological sex |
| `race` | Categorical | String | Standard NIH | Self-reported racial category |
| `bmi` | Float | $\text{kg/m}^2$ | 15.0 – 60.0 | Body Mass Index |
| `smoking_status` | Categorical | String | {'current', 'former', 'never'} | Current smoker vs former smoker (>6 months quit) |
| `pack_years` | Float | Pack-years | 0.0 – 180.0 | (Cigarettes per day / 20) $\times$ years smoked |
| `post_bd_fev1_l` | Float | Liters | 0.40 – 6.00 | Post-bronchodilator Forced Expiratory Volume in 1 second |
| `post_bd_fev1_pct_pred` | Float | % predicted | 15.0 – 140.0 | GLI 2012 global reference equations |
| `post_bd_fvc_l` | Float | Liters | 0.80 – 8.00 | Post-bronchodilator Forced Vital Capacity |
| `post_bd_fvc_pct_pred` | Float | % predicted | 25.0 – 140.0 | GLI 2012 global reference equations |
| `post_bd_fev1_fvc_ratio`| Float | Ratio | 0.15 – 0.95 | $\text{FEV}_1 / \text{FVC}$ post-bronchodilator |
| `gold_stage` | Integer | 0..4 | {0, 1, 2, 3, 4} | GOLD 0 ($\ge 0.70$), 1 ($\ge 80\%$), 2 ($50-79\%$), 3 ($30-49\%$), 4 ($<30\%$) |
| `prior_exacerbations_12m`| Integer | Count | 0 – 20 | Severe or moderate exacerbations in preceding 12 months |
| `mrc_dyspnea_scale` | Integer | 0..4 | {0, 1, 2, 3, 4} | Modified Medical Research Council dyspnea scale |
| `sgrq_total_score` | Float | Score | 0.0 – 100.0 | St. George's Respiratory Questionnaire total score |

---

## 3. Imaging Acquisition Specification

- **CT Scanner**: Multidetector helical CT ($\ge 64$ detector rows).
- **Breath-hold Protocols**:
  - **Inspiratory**: Maximum inspiration at Total Lung Capacity (TLC).
  - **Expiratory**: End of tidal expiration or residual volume (RV/FRC).
- **Reconstruction Parameters**:
  - Slice thickness: $\le 1.0$ mm (nominal $0.75-1.0$ mm).
  - Slice increment: $\le 0.7$ mm (overlapping slices preferred for registration).
  - Convolution kernel: Standard smooth reconstruction kernel for density analysis (e.g. Siemens B31f / GE Standard / Philips B). Avoid edge-enhancing/sharp kernels that artificially shift HU histograms.
  - Voxel physical units: Strictly calibrated Hounsfield Units with air calibration $-1000 \pm 10$ HU and water calibration $0 \pm 5$ HU.
