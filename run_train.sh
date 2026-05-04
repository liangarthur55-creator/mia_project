#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="mia-lightmedseg"

cd "$PROJECT_DIR"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Could not find conda. Please install Miniconda/Miniforge or open a terminal where conda is available."
  exit 1
fi

conda activate "$ENV_NAME"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-checkpoints/$RUN_ID}"
LOG_FILE="${LOG_FILE:-$RUN_DIR/train.log}"

mkdir -p "$RUN_DIR"
echo "Run directory: $RUN_DIR"
echo "Writing training log to $LOG_FILE"

python scripts/train_lightmedseg.py \
  --protocol pranet-light \
  --images-dir data/Kvasir-SEG/images \
  --masks-dir data/Kvasir-SEG/masks \
  --cvc-root data/CVC-ClinicDB \
  --output-dir "$RUN_DIR" \
  --best-metric dice \
  --variant tiny \
  --device mps \
  --image-size 352 \
  --size-rates 0.75,1,1.25 \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --val-ratio 0.1 \
  --save-split-dir data/splits/pranet_light_seed42 \
  "$@" 2>&1 | tee "$LOG_FILE"
