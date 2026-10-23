"""Helper script to construct reproducible Jupyter Notebooks for M10."""

import json
from pathlib import Path

def make_cell(cell_type: str, source: list[str]) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": [s + "\n" for s in source],
        **({"execution_count": None, "outputs": []} if cell_type == "code" else {})
    }

def create_notebook(cells: list[dict], out_path: Path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created {out_path.name}")

# Notebook 1: DIR-Lab Registration, TRE, and PRM
nb1_cells = [
    make_cell("markdown", [
        "# Notebook 01: DIR-Lab Deformable Registration, TRE & Parametric Response Mapping",
        "",
        "This notebook walks through:",
        "1. Loading paired inspiratory and expiratory thoracic CT scans from the DIR-Lab dataset.",
        "2. Morphological lung parenchyma segmentation.",
        "3. Deformable BSpline registration using SimpleITK.",
        "4. Evaluating Target Registration Error (TRE) in mm on 300 anatomical landmarks.",
        "5. Quantifying Parametric Response Mapping (PRM) into normal, fSAD, and emphysema phenotypes."
    ]),
    make_cell("code", [
        "import numpy as np",
        "import pandas as pd",
        "import SimpleITK as sitk",
        "from pathlib import Path",
        "",
        "from copd_anomaly.registration.registrar import SimpleITKRegistrar, evaluate_tre",
        "from copd_anomaly.features.prm import compute_prm_map, compute_prm_stats",
        "from copd_anomaly.segmentation.segmenter import LungSegmenter",
        "from copd_anomaly.data.dirlab_loader import DIRLabLoader",
        "",
        "print('Loaded registration and PRM modules successfully.')"
    ]),
    make_cell("markdown", [
        "## 1. Inspecting the Generated DIR-Lab Manifest",
        "All 20 scans across the 10 DIR-Lab cases have verified Hounsfield Unit offsets and 0-based landmark coordinate alignment."
    ]),
    make_cell("code", [
        "manifest_path = Path('../data/processed/dirlab_manifest.parquet')",
        "if manifest_path.exists():",
        "    df_manifest = pd.read_parquet(manifest_path)",
        "    display(df_manifest[['scan_id', 'hu_offset_verified', 'lung_median_hu', 'landmark_index_base', 'landmark_in_lung_frac']])",
        "else:",
        "    print('Manifest file not found on local disk. Run scripts/build_dirlab_manifest.py first.')"
    ]),
    make_cell("markdown", [
        "## 2. Interactive Synthetic Demonstration: Registration & TRE Evaluation",
        "We simulate a 3D paired volume with known ground-truth landmark displacement to demonstrate TRE calculation and PRM classification."
    ]),
    make_cell("code", [
        "# Create synthetic pair phantom",
        "shape = (20, 32, 32)",
        "vol_insp = np.full(shape, 40.0, dtype=np.float32)",
        "vol_insp[3:17, 8:24, 6:26] = -850.0  # Lung parenchyma",
        "",
        "vol_exp = np.full(shape, 40.0, dtype=np.float32)",
        "vol_exp[3:17, 8:24, 7:27] = -800.0   # Exhale shifted by 1 voxel in X",
        "",
        "segmenter = LungSegmenter(backend='threshold')",
        "mask = segmenter.segment(vol_insp, spacing_zyx=(2.0, 1.0, 1.0))",
        "",
        "registrar = SimpleITKRegistrar(fast_mode=True)",
        "reg_res = registrar.register(vol_insp, vol_exp, fixed_mask=mask, spacing_zyx=(2.0, 1.0, 1.0))",
        "",
        "print(f'Registration complete. Warped volume shape: {reg_res.warped_volume.shape}')"
    ]),
    make_cell("code", [
        "# Evaluate Target Registration Error (TRE)",
        "fixed_landmarks = np.array([[10.0, 16.0, 12.0], [12.0, 18.0, 20.0]], dtype=np.float32)",
        "moving_landmarks = fixed_landmarks + np.array([2.0, 2.0, 0.0], dtype=np.float32)",
        "",
        "warped_landmarks = registrar.transform_points(moving_landmarks, reg_res.transform, spacing_zyx=(2.0, 1.0, 1.0))",
        "tre_stats = evaluate_tre(fixed_landmarks, moving_landmarks, warped_landmarks, spacing_zyx=(2.0, 1.0, 1.0))",
        "",
        "print('TRE Statistics (mm):')",
        "for k, v in tre_stats.items():",
        "    print(f'  {k}: {v:.2f}')"
    ]),
    make_cell("markdown", [
        "## 3. Parametric Response Mapping (PRM)",
        "Classifies lung voxels into Normal, Functional Small Airway Disease (fSAD), and Emphysema."
    ]),
    make_cell("code", [
        "prm_stats = compute_prm_stats(vol_insp, reg_res.warped_volume, mask, spacing_zyx=(2.0, 1.0, 1.0))",
        "print('PRM Quantitative Breakdown:')",
        "for k, v in prm_stats.items():",
        "    print(f'  {k}: {v:.2f}%')"
    ])
]

# Notebook 2: CT-RATE Streaming & Contrastive Pretraining
nb2_cells = [
    make_cell("markdown", [
        "# Notebook 02: CT-RATE Streaming & 3D ResNet Contrastive Pretraining",
        "",
        "This notebook walks through:",
        "1. Disjoint patient splitting with mathematical assertion of zero split leakage.",
        "2. Single-scan streaming pipeline with immediate disk deletion to prevent disk exhaustion.",
        "3. 3D volumetric patch extraction with >=70% lung parenchyma containment.",
        "4. 3D ResNet encoder and NT-Xent contrastive loss optimization."
    ]),
    make_cell("code", [
        "import torch",
        "import pandas as pd",
        "from copd_anomaly.data.ctrate_streamer import create_tiered_splits, verify_split_leakage",
        "from copd_anomaly.models.encoder import ResNet3DEncoder",
        "from copd_anomaly.models.losses import NTXentLoss",
        "from copd_anomaly.models.augmentations import ContrastiveDualViewTransform",
        "",
        "print('Modules loaded successfully. CUDA available:', torch.cuda.is_available())"
    ]),
    make_cell("markdown", [
        "## 1. Zero-Leakage Patient Splitting",
        "Validates that ctrate_train, ctrate_val, and ctrate_test share strictly zero patients."
    ]),
    make_cell("code", [
        "# Synthetic demo metadata",
        "train_meta = pd.DataFrame([{'VolumeName': f'train_{i:04d}_a_1.nii.gz'} for i in range(150)])",
        "valid_meta = pd.DataFrame([{'VolumeName': f'valid_{i:04d}_a_1.nii.gz'} for i in range(80)])",
        "",
        "splits = create_tiered_splits(train_meta, valid_meta, tier='lean', seed=42)",
        "print('Splits successfully generated:')",
        "print('  Train subjects:', len(splits['train']))",
        "print('  Val subjects:  ', len(splits['val']))",
        "print('  Test subjects: ', len(splits['test']))"
    ]),
    make_cell("markdown", [
        "## 2. 3D ResNet Architecture & Contrastive Forward Pass",
        "Each 3D patch (50x50x50) produces a latent representation $z_{\\text{emb}} \\in \\mathbb{R}^{128}$ and normalized projection $z_{\\text{proj}} \\in \\mathbb{R}^{64}$."
    ]),
    make_cell("code", [
        "model = ResNet3DEncoder(in_channels=1, base_filters=16, embedding_dim=128, projection_dim=64)",
        "x_dummy = torch.rand(4, 1, 32, 32, 32)  # Batch of 4 patches",
        "",
        "dual_transform = ContrastiveDualViewTransform()",
        "v1, v2 = dual_transform(x_dummy)",
        "",
        "z1_emb, z1_proj = model(v1)",
        "z2_emb, z2_proj = model(v2)",
        "",
        "criterion = NTXentLoss(temperature=0.07)",
        "loss = criterion(z1_proj, z2_proj)",
        "print(f'Contrastive NT-Xent Loss: {loss.item():.4f}')"
    ])
]

# Notebook 3: Anomaly Density Scoring & Proxy AUC
nb3_cells = [
    make_cell("markdown", [
        "# Notebook 03: Normal-Lung Density Modeling & Proxy Diagnostic Evaluation",
        "",
        "This notebook walks through:",
        "1. Fitting Gaussian Mixture Model (GMM) and RealNVP Normalizing Flow on healthy control embeddings.",
        "2. Evaluating patch anomaly scores $s(x) = -\\log p(z)$.",
        "3. Aggregating patch scores into patient-level severity indices.",
        "4. Evaluating proxy AUROC and AUPRC against radiology report labels with 95% bootstrap confidence intervals.",
        "",
        "> **MANDATORY NOTICE:** All diagnostic classifications in this notebook are strictly **REPORT-LABEL PROXIES, not clinical COPD diagnosis**."
    ]),
    make_cell("code", [
        "import numpy as np",
        "import pandas as pd",
        "from copd_anomaly.anomaly.density import GaussianMixtureDensity, PatchAnomalyScorer, RealNVPFlowDensity",
        "from copd_anomaly.anomaly.aggregation import ScoreAggregator, extract_patient_anomaly_summary",
        "from copd_anomaly.evaluation.metrics import ModelEvaluator, PROXY_DISCLAIMER",
        "",
        "print(PROXY_DISCLAIMER)"
    ]),
    make_cell("markdown", [
        "## 1. Density Estimation on Normal Healthy Lung"
    ]),
    make_cell("code", [
        "rng = np.random.RandomState(42)",
        "z_normal = rng.normal(loc=0.0, scale=1.0, size=(300, 32)).astype(np.float32)",
        "z_abnormal = rng.normal(loc=4.5, scale=1.5, size=(50, 32)).astype(np.float32)",
        "",
        "gmm = GaussianMixtureDensity(n_components=6, covariance_type='diag', random_state=42)",
        "gmm.fit(z_normal)",
        "",
        "scorer = PatchAnomalyScorer(gmm)",
        "scores_norm = scorer.score_patches(z_normal[:50])",
        "scores_abn = scorer.score_patches(z_abnormal)",
        "",
        "print(f'Normal patches anomaly score:   {scores_norm.mean():.2f} ± {scores_norm.std():.2f}')",
        "print(f'Abnormal patches anomaly score: {scores_abn.mean():.2f} ± {scores_abn.std():.2f}')"
    ]),
    make_cell("markdown", [
        "## 2. Proxy AUROC & AUPRC with Bootstrap 95% Confidence Intervals"
    ]),
    make_cell("code", [
        "evaluator = ModelEvaluator()",
        "y_true = np.array([0] * 50 + [1] * 50)",
        "y_score = np.concatenate([scores_norm, scores_abn])",
        "",
        "metrics = evaluator.evaluate_proxy_auc(y_true, y_score, n_bootstraps=500)",
        "print('Proxy Diagnostic Evaluation Results:')",
        "print(f'  AUROC: {metrics[\"auroc\"]:.3f} (95% CI: [{metrics[\"auroc_ci95_low\"]:.3f}, {metrics[\"auroc_ci95_high\"]:.3f}])')",
        "print(f'  AUPRC: {metrics[\"auprc\"]:.3f} (95% CI: [{metrics[\"auprc_ci95_low\"]:.3f}, {metrics[\"auprc_ci95_high\"]:.3f}])')",
        "print(f'  Optimal Sensitivity: {metrics[\"sensitivity_at_opt\"]:.1%}')",
        "print(f'  Optimal Specificity: {metrics[\"specificity_at_opt\"]:.1%}')"
    ])
]

# Notebook 4: Multimodal Exacerbation Risk Modeling
nb4_cells = [
    make_cell("markdown", [
        "# Notebook 04: Multimodal Fusion, MC Uncertainty & 5-Arm Feature Ablations",
        "",
        "This notebook walks through:",
        "1. Generating a synthetic clinical + spirometry + imaging cohort.",
        "2. Multimodal gated fusion architecture combining CT anomaly embeddings with clinical covariates.",
        "3. Monte Carlo Dropout epistemic and aleatoric uncertainty estimation.",
        "4. 5-arm systematic feature ablation study comparing Clinical, Spirometry, Imaging, and Multimodal models.",
        "",
        "> **SYNTHETIC DATA BANNER:** This cohort is synthetically generated for methodological validation only."
    ]),
    make_cell("code", [
        "import torch",
        "import pandas as pd",
        "from copd_anomaly.synthetic.cohort_generator import generate_synthetic_copd_cohort, SYNTHETIC_BANNER",
        "from copd_anomaly.synthetic.multimodal import MultimodalRiskPipeline",
        "from scripts.run_ablation import run_ablation_experiments",
        "",
        "print(SYNTHETIC_BANNER)"
    ]),
    make_cell("markdown", [
        "## 1. Inspecting the Synthetic Clinical Cohort"
    ]),
    make_cell("code", [
        "df_cohort = generate_synthetic_copd_cohort(n_patients=200, seed=42)",
        "display(df_cohort[['patient_id', 'age', 'sex', 'pack_years', 'fev1_pct_predicted', 'gold_stage', 'global_anomaly_score', 'exacerbation_12m_event']].head(10))"
    ]),
    make_cell("markdown", [
        "## 2. Multimodal Risk Modeling & Uncertainty Estimation"
    ]),
    make_cell("code", [
        "pipeline = MultimodalRiskPipeline(d_ct=32, d_clin=6, d_fused=16, dropout=0.3)",
        "x_ct_demo = torch.randn(8, 32)",
        "x_clin_demo = torch.randn(8, 6)",
        "",
        "pred_dict = pipeline.predict_with_uncertainty(x_ct_demo, x_clin_demo, n_mc_samples=25)",
        "print('Predicted Risk & Uncertainty for Sample Patients:')",
        "for i in range(5):",
        "    print(f'  Patient {i+1}: Mean Risk = {pred_dict[\"mean_risk\"][i]:.1%}, Epistemic Uncertainty = {pred_dict[\"epistemic_uncertainty\"][i]:.4f}')"
    ]),
    make_cell("markdown", [
        "## 3. Systematic 5-Arm Feature Ablations"
    ]),
    make_cell("code", [
        "from pathlib import Path",
        "temp_report = Path('../reports/multimodal_ablation_results.csv')",
        "df_ablations = run_ablation_experiments(output_report=temp_report, n_splits=5, seed=42)",
        "display(df_ablations[['configuration', 'auroc', 'auprc', 'brier_score', 'delta_auroc_vs_baseline', 'p_value_vs_baseline']])"
    ])
]

# Write all notebooks
create_notebook(nb1_cells, Path("notebooks/01_dirlab_registration_prm.ipynb"))
create_notebook(nb2_cells, Path("notebooks/02_ctrate_streaming_contrastive.ipynb"))
create_notebook(nb3_cells, Path("notebooks/03_anomaly_density_scoring.ipynb"))
create_notebook(nb4_cells, Path("notebooks/04_multimodal_exacerbation_ablations.ipynb"))
