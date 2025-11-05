#!/usr/bin/env bash
# Complete QwenVL Training Launch Script with Full Parameter Documentation

WORKER_NODES=2
RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}

echo RANK: ${RANK}
echo WORKER_NODES: ${WORKER_NODES}
echo WORLD_SIZE: ${WORLD_SIZE}
echo MASTER_ADDR: ${MASTER_ADDR}
echo MASTER_PORT: ${MASTER_PORT}

# Re-run the script with Bash if it was invoked via another shell.
if [ -z "${BASH_VERSION:-}" ]; then
    exec /usr/bin/env bash "$0" "$@"
fi

set -euo pipefail

. .venv/bin/activate

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}          # Master node IP for multi-GPU training
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}  # Random port to avoid conflicts
if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l) # Automatically detects available GPUs
else
    echo "nvidia-smi not found; defaulting to 1 process per node." >&2
    NPROC_PER_NODE=1
fi

# ======================
# Path Configuration
# ======================
MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"           # Pretrained model path
OUTPUT_DIR="./checkpoints"                       # Directory for saving checkpoints
CACHE_DIR="./cache"                              # Cache directory for models
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${ROOT_DIR}/qwen-vl-finetune/qwenvl/train/train_qwen.py"
export PYTHONPATH="${ROOT_DIR}/qwen-vl-finetune:${PYTHONPATH:-}"
DATA_CONFIG_PATH="${SCRIPT_DIR}/data.json"

AOSS_CONF_PATH=${AOSS_CONF_PATH:-"/mnt/aigc/users/pufanyi/aoss.conf"}
AOSS_CONF_RULES=${AOSS_CONF_RULES:-"^s3://public-dataset/::/mnt/aigc/caizhongang/aoss.conf"}
MEDIA_CACHE_DIR=${MEDIA_CACHE_DIR:-"${CACHE_DIR}/media"}
export AOSS_CONF_PATH AOSS_CONF_RULES MEDIA_CACHE_DIR

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "Training entrypoint not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${DATA_CONFIG_PATH}" ]]; then
    echo "Dataset config not found: ${DATA_CONFIG_PATH}" >&2
    exit 1
fi

export QWENVL_DATA_CONFIG="${DATA_CONFIG_PATH}"
export PYTHONPATH="${ROOT_DIR}/qwen-vl-finetune:${PYTHONPATH:-}"

# ======================
# Model Configuration
# ======================
echo "Using training script: ${TRAIN_SCRIPT}"
echo "Using data config: ${DATA_CONFIG_PATH}"
DATASETS=$(python3 - "${DATA_CONFIG_PATH}" <<'PY'
import json
import sys

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset_entries = []
for name, cfg in data.items():
    repeat_time = cfg.get("repeat_time", 1)
    try:
        repeat_value = float(repeat_time)
    except (TypeError, ValueError):
        repeat_value = 1.0

    repeat_value = max(repeat_value, 0.0)
    if repeat_value == 0.0:
        continue

    percent = int(round(repeat_value * 100))
    # Ensure at least 1% when repeat_time is a small positive number.
    percent = max(percent, 1)
    dataset_entries.append(f"{name}%{percent}")

print(",".join(dataset_entries))
PY
)

echo "DATASETS: ${DATASETS}"

if [[ -z "${DATASETS}" ]]; then
    echo "No datasets resolved from ${DATA_CONFIG_PATH}" >&2
    exit 1
fi

# ======================
# Training Hyperparameters
# ======================
torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nnodes="${WORKER_NODES}" \
    --node_rank="${RANK}" \
    "${TRAIN_SCRIPT}" \
    --model_name_or_path "${MODEL_PATH}" \
    --tune_mm_llm True \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --dataset_use "${DATASETS}" \
    --output_dir "${OUTPUT_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --aoss_conf_path "${AOSS_CONF_PATH}" \
    --aoss_conf_rules "${AOSS_CONF_RULES}" \
    --media_cache_dir "${MEDIA_CACHE_DIR}" \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --optim adamw_torch \
    --model_max_length 4096 \
    --data_flatten True \
    --data_packing False \
    --max_pixels $((576*28*28)) \
    --min_pixels $((16*28*28)) \
    --video_fps 2 \
    --video_max_frames 16 \
    --video_min_frames 4 \
    --video_max_pixels $((1664*28*28)) \
    --video_min_pixels $((256*28*28)) \
    --num_train_epochs 1 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 100

#  # LoRA Config (disabled by default)
#  --lora_enable True \
#  --lora_r 8 \
#  --lora_alpha 16 \
#  --lora_dropout 0.0

#  # Advanced Options
#  --deepspeed zero3.json
