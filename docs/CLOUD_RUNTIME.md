# Cloud runtime compatibility: Google Colab and Kaggle

The repository is a normal Python package with path-based command-line tools. Core processing must not depend on a Colab-only or Kaggle-only Python API. Notebook cells should configure paths and call the same tested scripts rather than duplicate model/data logic.

## Supported baseline

- Python 3.10, 3.11, or 3.12 (see `pyproject.toml`).
- GPU recommended for encoder training; CT preprocessing and QC can run on CPU if memory/time permit.
- Install with `python -m pip install -e ".[dev]"`.
- Keep raw scans, derived masks, patch shards, checkpoints, and patient tables outside the Git repository.
- All scripts accept filesystem paths. Use a writable local runtime disk for temporary processing and checkpoint writes; copy approved artifacts to persistent private storage after validation.

## Runtime preflight (both platforms)

Run in a notebook cell:

```python
import os, shutil, sys, torch
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Working directory:", os.getcwd())
```

Stop before GPU-only training if CUDA is unavailable or the preflight diagnostic fails.

## Google Colab

Colab supports mounting Google Drive. Use an approved private Drive location and avoid printing directory listings containing patient-identifying filenames in shared notebook output.

```python
from google.colab import drive
drive.mount("/content/drive")
```

Example path setup (replace the directory with the user's actual private data location):

```python
from pathlib import Path
DATA_ROOT = Path("/content/drive/MyDrive/copd_anomaly_data")
WORK_ROOT = Path("/content/copd-anomaly-work")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
```

Prefer local `/content` for intermediate shards/checkpoints during a run, then copy validated outputs to Drive. Frequent random writes to Drive can be slow or interrupted. Never make gated data publicly link-accessible to simplify mounting.

## Kaggle

Kaggle does not support Colab's `drive.mount()` API. Approved attached datasets are usually visible under `/kaggle/input/<dataset-name>/` (read-only); writable outputs belong under `/kaggle/working/`.

```python
from pathlib import Path
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/copd-anomaly-work")
WORK_ROOT.mkdir(parents=True, exist_ok=True)
print("Available input mount:", INPUT_ROOT.exists())
```

If the only approved source copy is in Google Drive and is not visible to Kaggle, use Colab or an approved private transfer process. Do not create a public Kaggle dataset containing CT-RATE, DIR-Lab, patient metadata, patch shards, or derived patient-level artifacts.

## Shared command-line workflow

From the repository root, use the same scripts on either platform:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
python scripts/diagnose_training.py --shards-dir PATH_TO_TRAIN_SHARDS --checkpoint-dir PATH_TO_CHECKPOINTS --report PATH_TO_PREFLIGHT.json
```

Use explicit absolute paths in notebook cells. Do not rely on the current working directory after a notebook restart. Do not put tokens in source code, notebook outputs, environment files committed to Git, or run reports.

## Required platform-neutrality rules for future notebooks

1. Notebook cells only mount/select storage, validate paths, print environment details, invoke repository scripts, and display QC summaries.
2. Model and data logic stays in `src/copd_anomaly/` or `scripts/`.
3. Every run records the Git commit SHA, package versions, configuration, selected scan/patient IDs, and output paths.
4. Real and synthetic outputs remain separate (`outputs/real/` and `outputs/synthetic/`).
5. A runtime restart must be recoverable from persisted manifests/checkpoints; do not depend on in-memory notebook state.
6. No code may assume a Hugging Face token or any credential exists. If gated access needs authentication, configure it through the runtime's private secret mechanism without echoing it; never request or commit credentials.
7. A notebook may not claim clinical performance from CT-RATE report-label proxies or from synthetic labels.

## Scientific gate

Platform compatibility only means the commands can run in the environment. It does not establish correct segmentation, registration, model convergence, held-out performance, or clinical validity. See `PROGRESS.md` and `docs/REAL_TRAINING_RUNBOOK.md` for those gates.
