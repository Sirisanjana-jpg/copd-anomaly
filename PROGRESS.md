# COPD-Anomaly: Project Progress & Verification Tracker

## Milestone Overview

| Milestone | Description | Status | Tests Passed | Review Stop |
|---|---|---|---|---|
| **M0** | Plan Artifact & Architectural Blueprint | ✅ Completed | N/A (Plan Approved) | Completed |
| **M1** | Project Scaffold, Dependencies & CI Pipeline | ✅ Completed | 7 passed (scaffold, interfaces, helpers, secrets) | No |
| **M2** | DIR-Lab Ingestion, Auditing & Parquet Manifest | ✅ Completed | 6 passed (ingestion, HU offset, base detection, manifest) | No |
| **M3** | Segmentation, Deformable Registration, TRE & PRM | ✅ Completed | 4 passed (segmenter, deformable reg, TRE, PRM) | **YES (Review Stop 1)** |
| **M4** | 3D Patch Sampling & Uint8 Sharded Datasets | ✅ Completed | 3 passed (uint8 scaling, patch constraints, sharded dataset) | No |
| **M5** | CT-RATE Selection & Streaming Pipeline | ✅ Completed | 5 passed (patient ID, zero leakage, tiered splits, normal labels) | No |
| **M6** | 3D ResNet Contrastive Encoder & NT-Xent | ✅ Completed | 4 passed (3D ResNet, dual-channel, NT-Xent, dual-view augmentations) | No |
| **M7** | Density Estimation, Anomaly Scoring & Proxy AUC | ✅ Completed | 4 passed (GMM, RealNVP flow, aggregators, proxy AUROC) | **YES (Review Stop 2)** |
| **M8** | DIR-Lab Lobe-Wise Quantitative Analysis | 🔄 In Progress | Pending | No |
| **M9** | Multimodal Extension & Synthetic Cohort Ablations | ⏳ Pending | Pending | No |
| **M10** | Comprehensive Documentation, Notebooks & GitHub Ready | ⏳ Pending | Pending | **YES (Review Stop 3)** |

---

## Verification Log (Verified vs. Unverified)

### Verified Items
- [x] **V1**: Workspace path and local OS verified: Windows 11, PowerShell, local Python 3.11.8.
- [x] **V2**: `data/raw_zips/` verified: all 10 DIR-Lab archives (`copd1.zip` through `copd10.zip`) exist on local disk.
- [x] **V3**: Ground rules codified in `.agent/rules/project.md` and committed.
- [x] **V4**: Security guard configured: pre-commit secret regex rejects `hf_[A-Za-z0-9]{20,}` patterns.
- [x] **V5**: Architecture, formal protocols, and tensor shapes approved in M0 Master Plan.
- [x] **V6 (U1)**: DIR-Lab Landmark Index Base (0 vs 1): Verified at runtime by `dirlab_loader.py` by computing in-lung landmark percentage for both coordinate bases; logs result and enforces $\ge 90\%$ containment. Base 0 selected for all cases with >90% lung containment.
- [x] **V7 (U2)**: DIR-Lab copd2 Dimensions: Verified by `scripts/audit_zips.py` file size $53,477,376$ bytes confirming $512 \times 512 \times 102$.
- [x] **V8 (U3)**: DIR-Lab copd2-copd10 Internal Layout: Non-destructively verified by `scripts/audit_zips.py`, exactly matching primary dims.
- [x] **V9 (U4)**: DIR-Lab copd1-copd10 HU Offset: Verified across all 20 scans with median lung HU strictly $\in [-950, -750]$ HU.
- [x] **V10 (U5/U7)**: CT-RATE Patient ID parsing, zero-leakage disjoint splitting (`ctrate_train`, `ctrate_val`, `ctrate_test`), and canonical `_a_1.nii.gz` suffix preference verified.
- [x] **V11 (U6)**: CT-RATE Voxel Intensity Rescaling: Window [-1024, 0] HU to [0, 255] uint8 and [0, 1] float32 verified.
- [x] **V12 (U9)**: Density Modeling & Normalizing Flow: GMM density estimator and RealNVP normalizing flow implemented and verified with anomaly scoring $s(x) = -\log p(z)$.

### Unverified Items & Runtime Verification Plan
- [ ] **U8: Lobe Segmentation Backend & License**: Handled by dynamic probe with fallback to classical threshold mask and loud alert disabling lobe features if unavailable.

---

## Open Questions Log

1. **Target Python Version**: Local workspace is Python 3.11.8. Pinning `>=3.10,<3.12` in `pyproject.toml` ensures compatibility across Colab/Kaggle.
2. **Hugging Face Shards Repo**: User input pending for `HF_SHARDS_REPO` (e.g. `username/ctrate-patch-shards`).
3. **Local GPU**: Assumed local CPU-only; GPU jobs scheduled for Kaggle/Colab.
4. **Software License Choice**: Tracked as `LICENSE` TODO pending CT-RATE (CC-BY-NC-SA) and DIR-Lab redistribution review.
5. **Other 9 DIR-Lab Zips Consistency**: Pending execution of `scripts/audit_zips.py` in M2.
6. **CT-RATE Remote Metadata & Columns**: Handled dynamically at runtime in M5.
7. **Lobe Segmentation Backend**: Dynamic fallback implemented in M3.
