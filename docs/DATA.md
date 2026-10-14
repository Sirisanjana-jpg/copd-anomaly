# Data Documentation: DIR-Lab & CT-RATE

## 1. DIR-Lab / Emory COPDgene Cohort (10 Cases)

### 1.1 Overview & Physical Characteristics
The DIR-Lab COPD dataset contains 10 thoracic CT cases from the COPDGene study, each provided with maximum inspiratory (iBHCT) and end-expiratory (eBHCT) breath-hold scans alongside 300 expert-annotated anatomical landmarks per phase.

- **Storage Format**: Headerless, raw binary files (`.img`), little-endian signed 16-bit integers (`<i2`).
- **Reshape Order**: `(Z, Y, X)` in NumPy where spatial dimensions are `[X(cols), Y(rows), Z(slices)]`.
- **Hounsfield Unit Conversion**: $\text{HU} = \text{stored\_value} - 1000$. Padding value $-2000 \to -1024$ HU (air).
- **Physical Coordinates**: Slice thickness is $2.5$ mm across all cases. In-plane spacing ranges from $0.586$ mm to $0.742$ mm.

### 1.2 Audited Dimensions & Benchmarks
All 10 archives in `data/raw_zips/` were audited and verified to match configured dimensions:

| Case ID | Inhale/Exhale Dims $(X, Y, Z)$ | In-Plane Spacing (mm) | Slice Thickness (mm) | Initial Displacement mm (Mean ± SD) | Observer Noise mm (Mean ± SD) | Lowest Published mm (Mean ± SD) |
|---|---|---|---|---|---|---|
| `copd1` | $512 \times 512 \times 121$ | $0.625 \times 0.625$ | $2.50$ | $25.90 \pm 11.57$ | $0.65 \pm 0.73$ | $0.71 \pm 0.81$ |
| `copd2` | $512 \times 512 \times 102$ | $0.645 \times 0.645$ | $2.50$ | $21.77 \pm 6.46$ | $1.06 \pm 1.51$ | $1.46 \pm 2.28$ |
| `copd3` | $512 \times 512 \times 126$ | $0.652 \times 0.652$ | $2.50$ | $12.29 \pm 6.39$ | $0.58 \pm 0.87$ | $0.77 \pm 0.77$ |
| `copd4` | $512 \times 512 \times 126$ | $0.590 \pm 0.590$ | $2.50$ | $30.90 \pm 13.49$ | $0.71 \pm 0.96$ | At observer threshold |
| `copd5` | $512 \times 512 \times 131$ | $0.647 \times 0.647$ | $2.50$ | $30.90 \pm 14.05$ | $0.65 \pm 0.87$ | $0.71 \pm 0.83$ |
| `copd6` | $512 \times 512 \times 119$ | $0.633 \times 0.633$ | $2.50$ | $28.32 \pm 9.20$ | $1.06 \pm 2.38$ | At observer threshold |
| `copd7` | $512 \times 512 \times 112$ | $0.625 \times 0.625$ | $2.50$ | $21.66 \pm 7.66$ | $0.65 \pm 0.78$ | $0.74 \pm 1.06$ |
| `copd8` | $512 \times 512 \times 115$ | $0.586 \times 0.586$ | $2.50$ | $25.57 \pm 13.61$ | $0.96 \pm 3.07$ | At observer threshold |
| `copd9` | $512 \times 512 \times 116$ | $0.664 \times 0.664$ | $2.50$ | $14.84 \pm 10.01$ | $1.01 \pm 2.54$ | At observer threshold |
| `copd10`| $512 \times 512 \times 135$ | $0.742 \times 0.742$ | $2.50$ | $22.48 \pm 10.64$ | $0.87 \pm 1.65$ | At observer threshold |

*Note on `copd2`*: The DIR-Lab website notes potential dimensions of $256 \times 256 \times 112$ or $512 \times 512 \times 102$. Audit of raw archive confirmed actual file size is $53,477,376$ bytes, conclusively verifying $512 \times 512 \times 102$.

### 1.3 Landmark Coordinate System
- Coordinate columns: `x` (column index), `y` (row index), `z` (slice index).
- Index Base: Automated runtime detection tests both 0-based and 1-based coordinates against the segmented lung parenchyma mask. DIR-Lab coordinates follow 1-based indexing ($1 \le x, y \le 512, 1 \le z \le Z$); subtracting 1 aligns $100\%$ of inhale landmarks within the lung parenchyma.

---

## 2. CT-RATE Dataset (Chest CT with Radiology Text Reports)

### 2.1 Overview & Access
- **Source**: Hugging Face dataset repository `ibrahimhamamci/CT-RATE` (gated).
- **Scale**: ~21.3 TB across 50,188 volumes (~2.4 reconstructions per subject).
- **Modality**: Non-contrast 3D chest CT with corresponding report-extracted abnormality classifications.
- **Licence**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

### 2.2 Streaming & Local Footprint Control
To prevent disk exhaustion, CT-RATE volumes are **never downloaded in bulk**:
1. Candidate subjects are selected offline using metadata CSVs (`train_predicted_labels.csv` and `valid_predicted_labels.csv`).
2. Single volume is fetched via `huggingface_hub.hf_hub_download`.
3. Volume is resampled to $1.0$ mm isotropic spacing, $50 \times 50 \times 50$ patches are extracted, and patches are appended to a sharded binary dataset.
4. **The raw NIfTI volume is immediately deleted from disk**.
5. Completion is checkpointed in `selection.csv`.

### 2.3 Abnormality Labels & Normal Control Filtering
Normal control scans for training density models $p(z)$ require:
1. Zero positive classifications across all target thoracic abnormalities:
   - `Emphysema` = 0
   - `Mosaic attenuation pattern` = 0
   - `Lung opacity` = 0
   - `Consolidation` = 0
   - `Atelectasis` = 0
   - `Pulmonary fibrotic sequela` = 0
   - `Bronchiectasis` = 0
2. Quantitative patch emphysema fraction $< 0.01$ (voxels $< -950$ HU).
