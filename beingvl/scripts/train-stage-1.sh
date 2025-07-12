#!/bin/bash

# Being-VL Stage 1 Training: Embedding Alignment
# Purpose: Align newly added visual token embeddings with pre-trained language model
# Strategy: Freeze all parameters except new visual embeddings, use PT data

set -euo pipefail

# NCCL settings
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA="mlx5"

# Stage 1 Training Parameters
BS=1
GRAD_ACCUM=2
EPOCH=2
LR=1e-3
LR_SCHE="cosine"
WARMUP=0.03
TIME=$(date "+%Y%m%d%H%M%S")
MODEL_SIZE="8b"

TRAIN_TYPE="stage-1"
SAVE_STEPS=500000
TRAIN_FILE="beingvl/train/train.py"

# Master node ID
if [ $# -eq 0 ]; then
    echo "Error: Master node ID required"
    echo "Usage: $0 <master_node_id>"
    exit 1
fi
MASTER=$1

CONFIG_PATH="beingvl/scripts/accelerate/config-${MASTER}.yaml"

# Model Configuration
MODEL_TYPE="llama3+vqgan+8k"
TRAIN_WITH_VBPE="True"

# Stage 1 Model Path - Should be the initialized base model after convert_llama_to_being.py
# Replace with your actual workspace path
MODEL_PATH="/path/to/your/workspace/models/beingvl/base"

# Stage 1 Data - Pretraining data (Foundation Data 80% + Perception Data 20%)
# Replace with your actual workspace path - use the vBPE tokenized version
DATA_PATH="/path/to/your/workspace/data/tokenized/pt/pretrain_data_vbpe.jsonl"
SHUFFLE_DATA="True"

# Stage 1 Specific Settings
FREEZE_TEXT_HEADS="True"     # Freeze all parameters except embeddings
INIT_IMAGE_HEADS="True"      # Initialize new visual token embeddings

# Output directories - consistent workspace structure
OUTPUT_DIR="/path/to/your/workspace/models/beingvl/stage-1"
LOG_DIR="/path/to/your/workspace/logs/stage-1"

# Backup the training script
SCRIPT_PATH=$(readlink -f "$0")
mkdir -p "$OUTPUT_DIR"
cp "$SCRIPT_PATH" "$OUTPUT_DIR"
echo "Stage 1 training script backed up to: $OUTPUT_DIR"

echo "========================================="
echo "Being-VL Stage 1 Training: Embedding Alignment"
echo "Model Type: $MODEL_TYPE"
echo "Freeze Text Heads: $FREEZE_TEXT_HEADS"
echo "Initialize Image Heads: $INIT_IMAGE_HEADS"
echo "Learning Rate: $LR"
echo "Epochs: $EPOCH"
echo "Batch Size: $BS, Grad Accumulation: $GRAD_ACCUM"
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "========================================="

# Start Stage 1 Training
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

echo "Stage 1 training completed. Model saved to: $OUTPUT_DIR/final"
echo "Next: Run train-stage-2.sh with MODEL_PATH=$OUTPUT_DIR/final"