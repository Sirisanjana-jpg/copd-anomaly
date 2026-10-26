# DIR-Lab end-to-end exploratory run

The repository does not contain the raw DIR-Lab ZIPs. Keep permitted archives locally under `data/`; the pipeline discovers them recursively and does not upload or commit them.

## Prepare the local data

Put one archive per case somewhere under `data/`. Archive names must contain the case ID (for example, `copd1.zip` through `copd10.zip`). Each archive must contain the DIR-Lab inspiratory and expiratory raw volumes named with `_iBHCT*.img` and `_eBHCT*.img`. Landmark files are used when present. The code checks archive integrity and required volume members before processing.

Install the project and the optional five-lobe segmentation backend:

```powershell
python -m pip install -e ".[dev,segmentation]"
```

TotalSegmentator may require additional model downloads, and segmentation can be slow. Confirm PyTorch and CUDA availability before a long run:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Run extraction, patching, training and scoring

From the repository root:

```powershell
python scripts/run_dirlab_pipeline.py --data-dir data --output-root data/processed/dirlab_run --patch-size 50 --max-patches 100 --epochs 5 --batch-size 2 --components 4 --device auto
```

If you require a GPU rather than allowing the training script to use CPU:

```powershell
python scripts/run_dirlab_pipeline.py --data-dir data --output-root data/processed/dirlab_run --epochs 5 --batch-size 2 --device cuda --require-cuda
```

Use a new `--output-root` for a fresh extraction if `patch_manifest.parquet` already exists. The script does not delete or overwrite raw archives. It temporarily indexes archives for the existing converter, converts raw scans to NIfTI, generates five-lobe masks, derives whole-lung masks from those labels, extracts patches, trains the contrastive encoder, fits a GMM to patches with LAA-950 fraction `<= 0.01`, scores patches, and writes lobe-level anomaly features.

## Expected outputs

- `data/processed/dirlab_run/nifti/`: converted inspiratory/expiratory NIfTI volumes.
- `data/processed/dirlab_run/dirlab_manifest.parquet`: scan metadata and HU checks.
- `data/processed/dirlab_run/lobes/` and `masks/`: anatomical lobe masks and their whole-lung union masks.
- `data/processed/dirlab_run/patches/`: sharded patch arrays and patch manifest.
- `outputs/real/dirlab_run/checkpoints/encoder_best.pt`: trained encoder checkpoint.
- `outputs/real/dirlab_run/checkpoints/density_model.joblib`: fitted GMM.
- `outputs/real/dirlab_run/reports/`: patch, scan and patient anomaly scores plus `lobe_features.csv`.

## Scientific limitations

This is a pipeline-validation/exploratory run, not a complete publication evaluation. DIR-Lab has only a small set of COPD cases. The script trains and scores on the same cases, so these scores are not held-out performance estimates. LAA-950 `<= 0.01` is a pseudo-normal heuristic, not verified healthy control status. Five-lobe masks require manual/quantitative quality control. The script does not perform deformable inhale/exhale registration or generate PRM statistics; paired-scan PRM must wait until registered expiratory volumes have been validated. Do not claim diagnostic accuracy, clinical validity, or generalization from these outputs.
