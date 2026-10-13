# Project Ground Rules & Operational Constraints

## 0. ROLE AND GROUND RULES
You are building a complete, GitHub-ready research codebase. Follow these rules for the whole task:
1. **No hallucination.** Never invent dataset facts, file paths, column names, URLs, library APIs, benchmark numbers, or results. Anything marked UNVERIFIED below must be checked at runtime by code (and logged), or listed as an open question. If you cannot verify something, say so in the plan and make it a config value with a loud validation check.
2. **No assumptions presented as facts.** Where I have not told you something, do not fill it in. Put it under "Open questions" in your plan and use a clearly named placeholder in config.
3. **No fabricated results.** You cannot run my real data (no data, no GPU in this workspace). Test everything on small synthetic fixtures. Never claim real-data performance.
4. **No secrets in the repo or in output.** Hugging Face tokens are read from environment variable `HF_TOKEN` (or `HF_READ` / `HF_WRITE` on Kaggle/Colab secrets). Never print, log, or commit them.
5. **Work in milestones.** Produce the plan artifact first and wait for my approval. Then build milestone by milestone; after each: run tests, update `PROGRESS.md`, make a git commit with a clear message. Stop and summarize at the end of milestones M3, M7 and M10 for my review.
6. Target OS for local development is **Windows (PowerShell)**; heavy jobs run on **Kaggle (primary) or Colab (backup)**. Use `pathlib`, no bash-only scripts, no Makefile dependency.

## 4. WHAT THIS DATA CAN AND CANNOT SUPPORT (enforce in code and README)
- CANNOT: COPD-vs-control AUC (needs spirometry), clinical mixed-effects analyses, exacerbation prediction on real data. **Disable real-data AUC/accuracy reporting unless a real label column exists.**
- CAN (real data): registration TRE on DIR-Lab landmarks; per-lobe volume and %LAA-950 / %LAA-856; PRM per lobe from registered pairs; lobe-wise anomaly maps; **proxy** patient-level AUC of the anomaly score against CT-RATE report-derived emphysema (and exploratory mosaic attenuation) labels — always labelled "REPORT-LABEL PROXY, not COPD classification"; embedding/domain-shift diagnostics.
- CAN (synthetic only, always bannered): fusion, exacerbation risk heads, uncertainty, ablations, statistical tests.

## 5. SPLITS AND ROLES (patient-level only; assert zero overlap in code and tests)
| Set | Source | Role |
|---|---|---|
| dirlab_all (10) | DIR-Lab | TRE, PRM, lobe analysis, qualitative anomaly maps. Never used for tuning. Any fitting on them uses leave-one-patient-out. |
| ctrate_train | CT-RATE official train split | encoder pretraining; normal-lung model fit (report-normal subset, patches <1% emphysema) |
| ctrate_val | CT-RATE official train split, disjoint from ctrate_train | choose aggregation, GMM K, thresholds |
| ctrate_test | CT-RATE official valid split | proxy AUC, evaluated once |

Subset tiers (nested: each tier is a prefix of one seeded shuffle, so tiers can be topped up without re-downloading):
- Active tier: `lean` (train: 100, val: 30, test: 70)
- Zero abnormality definition for normal controls: `[emphysema, mosaic_attenuation_pattern, lung_opacity, consolidation, atelectasis, pulmonary_fibrotic_sequela, bronchiectasis]`
- Patches: size 50, per_scan_train: 200, per_scan_eval: 100, min_lung_fraction: 0.7, window_hu: [-1024, 0], dtype: uint8, normal_fit_emphysema_cap: 0.01, shard_size: 1000
- Harmonize: spacing_mm: 1.0, expected lung median HU insp: [-950, -750]
