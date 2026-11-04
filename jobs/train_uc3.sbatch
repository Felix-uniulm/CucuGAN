#!/bin/bash
#SBATCH --job-name=cucugan-squares
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=09:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

# The workspace must contain:
#   <workspace>/squares128/train/{A,B}
#   <workspace>/squares128/test/{A,B}
WS=$(ws_find cucugan)
DATA_NAME="squares128"
DATA_PATH="$TMPDIR/$DATA_NAME"
RUN="${RUN:-baseline_squares_2k}"
MAX_IMAGES="${MAX_IMAGES:-2000}"
SAVE_EVERY="${SAVE_EVERY:-10}"
TRAIN_EXTRA="${TRAIN_EXTRA:-}"

mkdir -p logs
mkdir -p "$WS/checkpoints" "$WS/outputs"
ln -sfn "$WS/checkpoints" checkpoints
ln -sfn "$WS/outputs" outputs

# Use the node-local SSD for image reads.
rm -rf "$DATA_PATH"
cp -r "$WS/$DATA_NAME" "$DATA_PATH"

echo "Job: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Run: $RUN"
echo "Data: $DATA_PATH"
echo "Max images: $MAX_IMAGES"
echo "Extra args: ${TRAIN_EXTRA:-<none>}"
echo "Start: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv

TRAIN_ARGS=(
    src/train.py
    --name "$RUN"
    --data "$DATA_PATH"
    --max-images "$MAX_IMAGES"
    --n-epochs 100
    --n-epochs-decay 100
    --batch-size 1
    --pool-size 50
    --device cuda
    --num-workers 6
    --save-every "$SAVE_EVERY"
)

if [[ -n "$TRAIN_EXTRA" ]]; then
    read -r -a EXTRA_ARGS <<< "$TRAIN_EXTRA"
    TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
fi

if [[ "${RESUME:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--resume)
fi

python -u "${TRAIN_ARGS[@]}"

echo "End: $(date --iso-8601=seconds)"
