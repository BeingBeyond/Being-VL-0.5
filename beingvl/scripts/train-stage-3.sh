#!/bin/bash

# Being-VL Stage 3 Training: Full Fine-tuning
# Purpose: Develop advanced reasoning and instruction-following capabilities
# Strategy: Unfreeze all parameters, use SFT data with instruction-heavy composition

set -euo pipefail

# NCCL settings
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA="mlx5"

# Stage 3 Training Parameters
BS=1
GRAD_ACCUM=4
EPOCH=3
LR=5e-5
LR_SCHE="cosine"
WARMUP=0.03
TIME=$(date "+%Y%m%d%H%M%S")
MODEL_SIZE="8b"

TRAIN_TYPE="stage-3"
SAVE_STEPS=500000
TRAIN_FILE="beingvl/train/train.py"

# Master node ID
if [ $# -eq 0 ]; then
    echo "Error: Master node ID required"
    echo "Usage: $0 <master_node_id> [stage2_model_path]"
    exit 1
fi
MASTER=$1

# Stage 2 model path (can be provided as second argument)
if [ $# -ge 2 ]; then
    STAGE2_MODEL_PATH="$2"
else
    # Default path - consistent workspace structure
    STAGE2_MODEL_PATH="/path/to/your/workspace/models/beingvl/stage-2/final"
fi

CONFIG_PATH="beingvl/scripts/accelerate/config-${MASTER}.yaml"

# Model Configuration - Fixed for llama3+vqgan+8k
MODEL_TYPE="llama3+vqgan+8k"
TRAIN_WITH_VBPE="True"

# Stage 3 Model Path - Should be the output from Stage 2
MODEL_PATH="$STAGE2_MODEL_PATH"

# Stage 3 Data - SFT data with instruction-heavy composition
# Data composition: Foundation Data 15% + Perception Data 15% + Reasoning Data 30% + Instruction Data 40%
# Replace with your actual workspace path - use the vBPE tokenized version
DATA_PATH="/path/to/your/workspace/data/tokenized/sft_stage3/sft_stage3_data_vbpe.jsonl"
SHUFFLE_DATA="True"

# Stage 3 Specific Settings
FREEZE_TEXT_HEADS="False"    # Unfreeze all parameters for full fine-tuning
INIT_IMAGE_HEADS="False"     # No initialization needed (already done in Stage 1)

# Output directories - consistent workspace structure
OUTPUT_DIR="/path/to/your/workspace/models/beingvl/stage-3"
LOG_DIR="/path/to/your/workspace/logs/stage-3"

# Backup the training script
SCRIPT_PATH=$(readlink -f "$0")
mkdir -p "$OUTPUT_DIR"
cp "$SCRIPT_PATH" "$OUTPUT_DIR"
echo "Stage 3 training script backed up to: $OUTPUT_DIR"

echo "========================================="
echo "Being-VL Stage 3 Training: Full Fine-tuning"
echo "Model Type: $MODEL_TYPE"
echo "Freeze Text Heads: $FREEZE_TEXT_HEADS"
echo "Initialize Image Heads: $INIT_IMAGE_HEADS"
echo "Learning Rate: $LR"
echo "Epochs: $EPOCH"
echo "Batch Size: $BS, Grad Accumulation: $GRAD_ACCUM"
echo "Stage 2 Model: $MODEL_PATH"
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "========================================="

# Verify Stage 2 model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Stage 2 model not found at $MODEL_PATH"
    echo "Please run Stage 2 training first or provide correct path"
    exit 1
fi

# Start Stage 3 Training
accelerate launch \
    --config_file "$CONFIG_PATH" \
    "$TRAIN_FILE" \
    --model_name_or_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --data_path "$DATA_PATH" \
    --shuffle_data "$SHUFFLE_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --freeze_text_heads "$FREEZE_TEXT_HEADS" \
    --init_image_heads "$INIT_IMAGE_HEADS" \
    --train_with_vbpe "$TRAIN_WITH_VBPE" \
    --num_train_epochs "$EPOCH" \
    --per_device_train_batch_size "$BS" \
    --bf16 "True" \
    --tf32 "True" \
    --save_strategy "steps" \
    --save_steps "$SAVE_STEPS" \
    --learning_rate "$LR" \
    --weight_decay 0. \
    --warmup_ratio "$WARMUP" \
    --lr_scheduler_type "$LR_SCHE" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --gradient_checkpointing "True" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --report_to "tensorboard" \
    --logging_dir "$LOG_DIR" \
    --resume_from_checkpoint "False"

echo "========================================="
echo "Stage 3 training completed! Final model saved to: $OUTPUT_DIR/final"
echo "Being-VL 3-stage training pipeline completed successfully!"
echo "Final model ready for inference and evaluation."
echo "========================================="