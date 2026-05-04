# MIA Project

LightMedSeg training and evaluation utilities for medical image segmentation experiments.

## Project Structure

- `models/`: model definitions.
- `scripts/`: training and evaluation entrypoints.
- `tests/`: smoke tests for the LightMedSeg pipeline.
- `run_train.sh`: default training launcher.
- `run_eval.sh`: evaluation launcher.
- `run_tests.sh`: test launcher.
- `data/splits/`: small saved split files used by experiments.

Large datasets, checkpoints, logs, and third-party reference clones are intentionally excluded from Git.

## Setup

Create and activate the expected Conda environment, then install dependencies:

```bash
conda create -n mia-lightmedseg python=3.10
conda activate mia-lightmedseg
pip install -r requirements.txt
```

## Run

```bash
./run_tests.sh
./run_train.sh
./run_eval.sh
```

The default training script expects local datasets under `data/`, including `data/Kvasir-SEG` and `data/CVC-ClinicDB`.
