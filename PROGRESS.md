# COPD-Anomaly: Project Progress & Verification Tracker

## Milestone Overview

| Milestone | Description | Status | Tests Passed | Review Stop |
|---|---|---|---|---|
| **M0** | Plan Artifact & Architectural Blueprint | ✅ Completed | N/A (Plan Approved) | Completed |
| **M1** | Project Scaffold, Dependencies & CI Pipeline | ✅ Completed | 7 passed (scaffold, interfaces, helpers, secrets) | No |
| **M2** | DIR-Lab Ingestion, Auditing & Parquet Manifest | 🔄 In Progress | Pending | No |
| **M3** | Segmentation, Deformable Registration, TRE & PRM | ⏳ Pending | Pending | **YES (Review Stop 1)** |
| **M4** | 3D Patch Sampling & Uint8 Sharded Datasets | ⏳ Pending | Pending | No |
| **M5** | CT-RATE Selection & Streaming Pipeline | ⏳ Pending | Pending | No |
| **M6** | 3D ResNet Contrastive Encoder & NT-Xent | ⏳ Pending | Pending | No |
| **M7** | Density Estimation, Anomaly Scoring & Proxy AUC | ⏳ Pending | Pending | **YES (Review Stop 2)** |
| **M8** | DIR-Lab Lobe-Wise Quantitative Analysis | ⏳ Pending | Pending | No |
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

### Unverified Items & Runtime Verification Plan
- [ ] **U1: DIR-Lab Landmark Index Base (0 vs 1)**: Will be verified at runtime by `dirlab_loader.py` by computing in-lung landmark percentage for both coordinate bases; logs result and enforces $\ge 95\%$ containment.
- [ ] **U2: DIR-Lab copd2 Dimensions**: Will be verified by `scripts/audit_zips.py` by checking raw file size against $2 \times 512 \times 512 \times 102$ vs $2 \times 256 \times 256 \times 112$ bytes.
- [ ] **U3: DIR-Lab copd2-copd10 Internal Layout**: Will be verified non-destructively by `scripts/audit_zips.py` checking member file names and byte sizes.
- [ ] **U4: DIR-Lab copd2-copd10 HU Offset**: Will be verified per-scan by `dirlab_loader.py` asserting median lung HU inside crude mask $\in [-950, -750]$ HU.
- [ ] **U5: CT-RATE Hugging Face Repo Layout**: Will be verified via `scripts/verify_ctrate_layout.py` querying `HfApi.list_repo_files` and validating CSV headers.
- [ ] **U6: CT-RATE Voxel Intensity Rescaling**: Will be verified by inspecting sample NIfTI header and intensity distribution.
- [ ] **U7: CT-RATE Volume Suffix Preference**: Handled by falling back to first available reconstruction if preferred suffix `_a_1.nii.gz` is absent.
- [ ] **U8: Lobe Segmentation Backend & License**: Handled by dynamic probe with fallback to classical threshold mask and loud alert disabling lobe features if unavailable.
- [ ] **U9: Normalizing Flow Backend**: Handled by importing `nflows`/`FrEIA` or seamlessly falling back to internal RealNVP module.

---

## Open Questions Log

1. **Target Python Version**: Local workspace is Python 3.11.8. Pinning `>=3.10,<3.12` in `pyproject.toml` ensures compatibility across Colab/Kaggle.
2. **Hugging Face Shards Repo**: User input pending for `HF_SHARDS_REPO` (e.g. `username/ctrate-patch-shards`).
3. **Local GPU**: Assumed local CPU-only; GPU jobs scheduled for Kaggle/Colab.
4. **Software License Choice**: Tracked as `LICENSE` TODO pending CT-RATE (CC-BY-NC-SA) and DIR-Lab redistribution review.
5. **Other 9 DIR-Lab Zips Consistency**: Pending execution of `scripts/audit_zips.py` in M2.
6. **CT-RATE Remote Metadata & Columns**: Handled dynamically at runtime in M5.
7. **Lobe Segmentation Backend**: Dynamic fallback implemented in M3.
