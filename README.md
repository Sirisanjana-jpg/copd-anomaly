# COPD-Anomaly: Deep Anomaly-Based Assessment of Chest CT & Lobe-Wise Exacerbation Risk

[![CI](https://github.com/Sirisanjana-jpg/copd-anomaly/actions/workflows/ci.yml/badge.svg)](https://github.com/Sirisanjana-jpg/copd-anomaly/actions/workflows/ci.yml)
[![Python 3.10 | 3.11](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: TODO](https://img.shields.io/badge/license-TODO-yellow.svg)](#license--terms)

A modular, reproducible implementation and extension of the self-supervised anomaly quantification method described by **Almeida et al. (2024)** (*"Prediction of disease severity in COPD: a deep learning approach for anomaly-based quantitative assessment of chest CT"*, *European Radiology*, method name **cOOpD**).

---

## 1. Scope & Research Objective

- **Base Method Reproduction**: Self-supervised 3D contrastive learning ($50 \times 50 \times 50$ lung parenchyma patches) trained on disease-unlabeled CT scans $\to$ latent space density modeling on normal controls $\to$ patch anomaly score $s(x) = -\log p(f(x))$ $\to$ subject-level aggregation $S(X)$.
- **Extension Research Question**: Do **lobe-wise anomaly features** derived from CT add incremental predictive value for **12-month exacerbation risk** beyond clinical variables and a global anomaly score?
- **Modality Scope**: Chest computed tomography (inspiratory $\pm$ registered expiratory) + clinical parameters.
- **Strict Exclusions**: Chest X-ray, MRI, and any other imaging modality.

> [!IMPORTANT]
> **Data Scope & Honest Reporting Boundaries**:
> - The available 10 DIR-Lab cases have **no clinical tables, no spirometry, no healthy controls, and no exacerbation labels**. They are used exclusively for physical registration evaluation (TRE), parametric response mapping (PRM), and qualitative anomaly visualizations.
> - CT-RATE labels are **report-derived text proxies**, not spirometry-confirmed COPD. Real-data AUC is reported strictly as a **REPORT-LABEL PROXY**, never as clinical COPD classification.
> - Exacerbation risk heads, multimodal fusion, and hypothesis tests (likelihood-ratio test, conformal intervals) are evaluated on a **synthetic cohort** with a documented planted effect, with mandatory banners on all outputs.

---

## 2. Repository Layout

```
copd-anomaly/
├── README.md                          # Project overview, scope, and instructions
├── PROGRESS.md                        # Milestone tracking, verification log, open questions
├── pyproject.toml                     # Python package specifications and pinned dependencies
├── .gitignore                         # Strict exclusion of data, weights, tokens, and outputs
├── .env.example                       # Template for HF_TOKEN and private repository settings
├── .pre-commit-config.yaml            # Secret leak scanning and linting hooks
├── .github/workflows/ci.yml           # Automated CPU-only testing on synthetic fixtures
├── .agent/rules/project.md            # Hard constraints and data rules
├── configs/                           # Modular Hydra/OmegaConf configurations
│   ├── config.yaml                    # Master experiment configuration
│   ├── data/                          # DIR-Lab, CT-RATE, and harmonization settings
│   ├── segmentation/                  # Pluggable lung and lobe segmentation backends
│   ├── patches/                       # 50^3 patch extraction parameters
│   ├── encoder/                       # 3D ResNet backbone and NT-Xent contrastive loss
│   ├── density/                       # GMM and Normalizing Flow density estimation
│   ├── aggregation/                   # Subject-level patch score aggregations
│   ├── fusion/                        # Multimodal early and late gated fusion
│   └── risk_head/                     # Exacerbation probability heads (BCE/Focal/Cox)
├── data/                              # Local data root (git-ignored)
│   ├── raw_zips/                      # DIR-Lab raw archives (copd1.zip - copd10.zip)
│   ├── interim/                       # Intermediate caches
│   └── processed/                     # NIfTI volumes, masks, registrations, patch shards
├── src/copd_anomaly/                  # Core Python source package
│   ├── interfaces.py                  # Formal Protocols and ABCs with exact tensor shapes
│   ├── data/                          # DIR-Lab loader, CT-RATE downloader, manifests
│   ├── preprocessing/                 # Resampling, windowing, and intensity validation
│   ├── segmentation/                  # Pluggable lobe segmenter with fallback
│   ├── registration/                  # SimpleITK deformable registration and TRE evaluator
│   ├── patches/                       # 3D patch extraction and lazy sharded dataset
│   ├── models/                        # 3D ResNet, projection head, NT-Xent loss
│   ├── anomaly/                       # GMM and RealNVP density models, anomaly scorer
│   ├── features/                      # Lobe feature builder, PRM, heterogeneity index
│   ├── labels/                        # Temporal 365-day exacerbation window constructor
│   ├── uncertainty/                   # Split-conformal prediction intervals
│   ├── evaluation/                    # TRE, proxy AUC, C-index, LRT, DCA metrics
│   ├── synthetic/                     # Synthetic cohort generator with planted effects
│   └── viz/                           # Coronal anomaly overlays, QC montages, heatmaps
├── scripts/                           # Runnable CLI pipelines (pathlib-based)
├── notebooks/                         # Kaggle and Colab execution notebooks
│   ├── kaggle_phaseA_cpu_stream.ipynb # Stream CT-RATE, harmonize, extract uint8 shards
│   ├── kaggle_phaseB_gpu_train.ipynb  # GPU 3D ResNet pretraining & density estimation
│   └── colab_backup.ipynb             # Backup Colab training notebook
├── tests/                             # Unit and integration test suite (CPU only)
└── docs/                              # In-depth architectural and schema documentation
    ├── ARCHITECTURE.md                # Flowcharts, design, and mathematical formulas
    ├── DATA.md                        # Data provenance, dimensions, and verification log
    ├── REAL_COHORT_SCHEMA.md          # Specification for future real-world clinical cohorts
    └── LIMITATIONS.md                 # Methodological and observational constraints
```

---

## 3. Quickstart Guide

### 3.1 Local Development (Windows PowerShell)

```powershell
# 1. Clone repository and navigate to workspace
git clone https://github.com/Sirisanjana-jpg/copd-anomaly.git
cd copd-anomaly

# 2. Set up virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev]"

# 3. Configure local environment variables
Copy-Item .env.example .env
# Edit .env and insert your HF_TOKEN

# 4. Install pre-commit hooks (verifies secrets and code style)
pre-commit install

# 5. Run test suite on CPU using synthetic fixtures
pytest tests/ -v
```

### 3.2 Running the DIR-Lab Pipeline Locally

```powershell
# Audit raw archives (validates file sizes without full extraction)
python scripts/audit_zips.py --zips-dir data/raw_zips

# Extract and build master manifest (checks HU offsets and landmark ordering)
python scripts/build_dirlab_manifest.py

# Run deformable registration and benchmark TRE
python scripts/run_registration.py --config-name experiment/dirlab_pipeline

# Compute parametric response mapping (PRM) and lobe statistics
python scripts/compute_prm.py
python scripts/lobe_analysis.py
```

### 3.3 CT-RATE Streaming & GPU Training

Cloud execution is supported through the same path-based scripts on **Google Colab or Kaggle**; see [`docs/CLOUD_RUNTIME.md`](docs/CLOUD_RUNTIME.md). The Kaggle real-data smoke checklist is in [`docs/KAGGLE_REAL_DATA_SMOKE.md`](docs/KAGGLE_REAL_DATA_SMOKE.md).


Use the repository's resumable streaming CLI to prepare patient-disjoint CT-RATE patch shards, then train on the training split only. See [`docs/REAL_TRAINING_RUNBOOK.md`](docs/REAL_TRAINING_RUNBOOK.md) for the gated-access setup, one-volume smoke run, split outputs, resumable training commands, and current limitations.

The streaming CLI uses threshold lung segmentation as a provisional baseline; inspect the masks and patch counts before a long GPU run. Real patient-level anomaly scoring and held-out evaluation are not yet implemented end-to-end. Do not treat successful pretraining or density fitting as evidence of diagnostic performance.

---

## 4. Current Experimental Status & Results Placeholders

| Milestone / Task | Status | Result / Metric | Note |
|---|---|---|---|
| **M1: Scaffold & CI** | In Progress | CI Workflow Configured | Synthetic CPU test suite |
| **M2: DIR-Lab Audit** | Pending | Not yet run | Pending audit of copd1-copd10 |
| **M3: DIR-Lab Registration** | Pending | Target: TRE $< 1.5$ mm | Observer noise floor: $\approx 0.65 - 1.06$ mm |
| **M5: CT-RATE Lean Subset** | Pending | 200 volumes selected | Streamed without local persistence |
| **M6: 3D Contrastive Encoder**| Pending | Not yet run | NT-Xent on `ctrate_train` |
| **M7: Proxy Emphysema AUC** | Pending | Baseline %LAA vs Anomaly | Bannered: `REPORT-LABEL PROXY` |
| **M9: 5-Model Ablation (Synthetic)**| Pending | LRT p-value | Validates planted effect |

*All results will be logged in `PROGRESS.md` as milestones are completed.*

---

## 5. License & Data Terms

- **Code License**: `TODO: Open-source license selection pending` (see review checklist).
- **Data Redistribution**:
  - **DIR-Lab**: Data provided by Emory University. Landmark and scan files are governed by DIR-Lab data use terms.
  - **CT-RATE**: Governed by Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
  - Derived datasets and pre-extracted patch shards cannot be publicly redistributed without adherence to these licenses.

---

## 6. Citation

If referencing this codebase or methodology, please cite the foundational paper:

```bibtex
@article{almeida2024prediction,
  title={Prediction of disease severity in COPD: a deep learning approach for anomaly-based quantitative assessment of chest CT},
  author={Almeida, et al.},
  journal={European Radiology},
  volume={34},
  pages={4379--4392},
  year={2024},
  publisher={Springer}
}
```
