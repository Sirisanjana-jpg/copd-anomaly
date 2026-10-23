# DIR-Lab & CT-RATE Dataset Verification and Audit Report

## 1. Executive Summary

A non-destructive audit of all 10 DIR-Lab COPD dataset archives (`copd1.zip` through `copd10.zip`) located in `data/raw_zips/` was conducted. All archive members, image headerless raw dimensions, byte sizes, and landmark files were verified against the configuration specification (`configs/data/dirlab_cases.yaml`).

- **Total Archives Audited**: 10
- **Total Compressed Size**: 1,141.73 MB (1.14 GB)
- **Total Uncompressed Size**: 1,203.14 MB (1.20 GB)
- **Raw Volume Images Size**: 1,203.00 MB
- **Physical Dimension Verification Status**: 100% Matched (0 Mismatches, 0 Corruptions)

---

## 2. DIR-Lab Case-by-Case Verification Matrix

| Case ID | Inhale Dims (X, Y, Z) | Exhale Dims (X, Y, Z) | Expected Bytes | Actual Bytes | Dimension Match Type | Spacing (sx, sy, sz) mm | Status |
|---|---|---|---|---|---|---|---|
| `copd1` | 512 × 512 × 121 | 512 × 512 × 121 | 63,438,848 | 63,438,848 | Primary | 0.625 × 0.625 × 2.50 | ✅ PASS |
| `copd2` | 512 × 512 × 102 | 512 × 512 × 102 | 53,477,376 | 53,477,376 | Primary (Resolved) | 0.645 × 0.645 × 2.50 | ✅ PASS |
| `copd3` | 512 × 512 × 126 | 512 × 512 × 126 | 66,060,288 | 66,060,288 | Primary | 0.652 × 0.652 × 2.50 | ✅ PASS |
| `copd4` | 512 × 512 × 126 | 512 × 512 × 126 | 66,060,288 | 66,060,288 | Primary | 0.590 × 0.590 × 2.50 | ✅ PASS |
| `copd5` | 512 × 512 × 131 | 512 × 512 × 131 | 68,681,728 | 68,681,728 | Primary | 0.647 × 0.647 × 2.50 | ✅ PASS |
| `copd6` | 512 × 512 × 119 | 512 × 512 × 119 | 62,390,272 | 62,390,272 | Primary | 0.633 × 0.633 × 2.50 | ✅ PASS |
| `copd7` | 512 × 512 × 112 | 512 × 512 × 112 | 58,720,256 | 58,720,256 | Primary | 0.625 × 0.625 × 2.50 | ✅ PASS |
| `copd8` | 512 × 512 × 115 | 512 × 512 × 115 | 60,293,120 | 60,293,120 | Primary | 0.586 × 0.586 × 2.50 | ✅ PASS |
| `copd9` | 512 × 512 × 116 | 512 × 512 × 116 | 60,817,408 | 60,817,408 | Primary | 0.664 × 0.664 × 2.50 | ✅ PASS |
| `copd10`| 512 × 512 × 135 | 512 × 512 × 135 | 70,778,880 | 70,778,880 | Primary | 0.742 × 0.742 × 2.50 | ✅ PASS |

---

## 3. Critical Edge Case Resolutions

### 3.1 `copd2` Ambiguity Resolved
The literature reports two candidate dimensions for `copd2`:
1. $256 \times 256 \times 112$ (14,680,064 bytes)
2. $512 \times 512 \times 102$ (53,477,376 bytes)

Our byte-level audit confirmed `copd2_iBHCT.img` and `copd2_eBHCT.img` are exactly $53,477,376$ bytes, conclusively validating the primary configuration $512 \times 512 \times 102$.

### 3.2 Hounsfield Unit Verification
- Headerless int16 conversion: $\text{HU} = \text{stored\_value} - 1000$
- Padding value (typically $-2000$) remapped to air ($-1024$ HU).
- Lung parenchyma median HU was verified to lie strictly within $[-950, -750]$ HU.

### 3.3 Landmark Index Base Detection
DIR-Lab landmark text files contain 300 coordinate triplets $(x, y, z)$. At runtime, coordinate containment within the lung parenchyma boundary is evaluated for both 0-based and 1-based hypotheses:
- Inhale landmarks: verified with $\ge 93\%$ containment under 0-based alignment (1-based text coordinates decremented by 1).
- Out-of-bounds voxels: 0%.

---

## 4. CT-RATE Remote Data Verification Protocol

- **Dataset Identifier**: `ibrahimhamamci/CT-RATE`
- **Streaming Policy**: Ingestion strictly streams scan-by-scan into sharded uint8 patch caches, deleting original NIfTI files immediately after extraction to ensure zero disk overflow.
- **Normal Filter**: Only scans annotated with 0 for all seven key pulmonary labels (emphysema, mosaic attenuation, lung opacity, consolidation, atelectasis, fibrotic sequelae, bronchiectasis) are admitted into the baseline density model training set.
