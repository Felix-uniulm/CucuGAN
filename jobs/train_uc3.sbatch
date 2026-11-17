#!/bin/bash
#SBATCH --job-name=cucugan-skin-full
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

# The workspace must contain:
#   <workspace>/squares128/train/{A,B}
#   <workspace>/squares128/test/{A,B}
WS=$(ws_find cucugan)
DATA_NAME="${DATA_NAME:-squares128}"
DATA_PATH="$TMPDIR/$DATA_NAME"
RUN="${RUN:-skin_ears_neck_tiles2_full}"
MAX_IMAGES="${MAX_IMAGES:-all}"
SAVE_EVERY="${SAVE_EVERY:-10}"
N_EPOCHS="${N_EPOCHS:-100}"
N_EPOCHS_DECAY="${N_EPOCHS_DECAY:-100}"
STRUCT_MASK="${STRUCT_MASK:-none}"
SKIN_LOSS="${SKIN_LOSS:-1}"
TEXTURE_TILES="${TEXTURE_TILES:-2}"
MASK_FOLDER="${MASK_FOLDER:-masks_A_skin_ears_neck}"
FACE_MODEL="${FACE_MODEL:-jonathandinu/face-parsing}"
TRAIN_EXTRA="${TRAIN_EXTRA:-}"

if [[ "$MAX_IMAGES" != "all" && ! "$MAX_IMAGES" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_IMAGES must be 'all' or a positive integer" >&2
    exit 1
fi

if [[ "${RESUME:-0}" != "1" && -e "$WS/checkpoints/$RUN/latest.pt" ]]; then
    echo "Run $RUN already exists. Use RESUME=1 or a new RUN name." >&2
    exit 1
fi

mkdir -p logs
mkdir -p "$WS/checkpoints" "$WS/outputs"
ln -sfn "$WS/checkpoints" checkpoints
ln -sfn "$WS/outputs" outputs

echo "Job: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Run: $RUN"
echo "Data: $DATA_PATH"
echo "Max images: $MAX_IMAGES"
echo "Epochs: $N_EPOCHS + $N_EPOCHS_DECAY decay"
echo "Structure mask: $STRUCT_MASK"
echo "Skin loss: $SKIN_LOSS; texture tiles: $TEXTURE_TILES"
echo "Mask folder: $MASK_FOLDER"
echo "Extra args: ${TRAIN_EXTRA:-<none>}"
echo "Start: $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv

if [[ "$SKIN_LOSS" == "1" ]]; then
    # Persist masks and model weights across jobs, including interrupted preparation.
    export HF_HOME="${HF_HOME:-$WS/huggingface}"
    echo "Preparing missing skin masks on GPU (model: $FACE_MODEL)"
    python -u scripts/prepare_masks.py \
        --data "$WS/$DATA_NAME" \
        --device cuda \
        --model "$FACE_MODEL" \
        --mask-folder "$MASK_FOLDER" \
        --threads "${SLURM_CPUS_PER_TASK:-4}"
fi

# Copy images together with the prepared masks to the node-local SSD.
rm -rf "$DATA_PATH"
cp -r "$WS/$DATA_NAME" "$DATA_PATH"

TRAIN_ARGS=(
    src/train.py
    --name "$RUN"
    --data "$DATA_PATH"
    --n-epochs "$N_EPOCHS"
    --n-epochs-decay "$N_EPOCHS_DECAY"
    --batch-size 1
    --pool-size 50
    --device cuda
    --num-workers 6
    --save-every "$SAVE_EVERY"
    --struct-mask "$STRUCT_MASK"
    --texture-tiles "$TEXTURE_TILES"
    --mask-folder "$MASK_FOLDER"
)

if [[ "$MAX_IMAGES" != "all" ]]; then
    TRAIN_ARGS+=(--max-images "$MAX_IMAGES")
fi

if [[ "$SKIN_LOSS" == "1" ]]; then
    TRAIN_ARGS+=(--skin-loss)
fi

if [[ -n "$TRAIN_EXTRA" ]]; then
    read -r -a EXTRA_ARGS <<< "$TRAIN_EXTRA"
    TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
fi

if [[ "${RESUME:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--resume)
fi

python -u "${TRAIN_ARGS[@]}"

echo "End: $(date --iso-8601=seconds)"
