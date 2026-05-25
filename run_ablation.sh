#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_MD="${RESULT_MD:-ablation_results.md}"

cd "$PROJECT_DIR"

echo "# LightMedSeg Ablation Results" > "$RESULT_MD"
echo "" >> "$RESULT_MD"
echo "Generated at: $(date)" >> "$RESULT_MD"
echo "" >> "$RESULT_MD"

for EXP in full no_lha no_egff no_dfc; do
  EXTRA_ARGS=()

  case "$EXP" in
    full)
      ;;
    no_lha)
      EXTRA_ARGS+=(--no-lha)
      ;;
    no_egff)
      EXTRA_ARGS+=(--no-egff)
      ;;
    no_dfc)
      EXTRA_ARGS+=(--no-dfc)
      ;;
  esac

  echo "## $EXP" >> "$RESULT_MD"
  echo '```text' >> "$RESULT_MD"
  echo "Training $EXP ..." | tee -a "$RESULT_MD"

  RUN_ID="ablation_$EXP" ./run_train.sh \
    --device cuda \
    --protocol pranet-light \
    --variant small \
    --image-size 352 \
    --batch-size 16 \
    --epochs 150 \
    --lr 5e-4 \
    --edge-loss-weight 0.3 \
    --num-workers 4 \
    --no-multi-scale \
    "${EXTRA_ARGS[@]}"

  echo "" >> "$RESULT_MD"
  echo "Evaluation $EXP ..." | tee -a "$RESULT_MD"

  ./run_eval.sh \
    --device cuda \
    --batch-size 16 \
    --checkpoint "checkpoints/ablation_$EXP/lightmedseg_best_dice.pt" \
    2>&1 | tee -a "$RESULT_MD"

  echo '```' >> "$RESULT_MD"
  echo "" >> "$RESULT_MD"
done
