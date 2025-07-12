# Training Guide

## Prerequisites

Before starting training, ensure you have completed:

1. **Environment Setup**: Install Being-VL and dependencies (see README.md)
2. **Data Preparation**: Complete all steps in [Data.md](Data.md)
   - VQ token extraction
   - vBPE tokenizer training
   - Dataset tokenization for all stages
3. **Model Initialization**: Convert LLaMA to Being-VL base model
4. We use `accelerate` for multi-node training. You may modify the training scripts to use your own framework like `slurm` or `deepspeed`, etc.

## Part 1: Model Initialization

### Step 1.1: Convert LLaMA to Being-VL Base Model

Initialize the base model from LLaMA 3.1-8B for training:

```bash
python beingvl/utils/convert_llama_to_being.py \
    --llama_path /path/to/your/workspace/models/llama3.1-8b \
    --being_tokenizer_config_path /path/to/your/workspace/models/being-tokenizer \
    --being_vq_path /path/to/your/workspace/models/being-vq-8k \
    --output_path /path/to/your/workspace/models/beingvl/base \
    --verify_loading
```

**Expected Output:**
- Initialized Being-VL base model in `/path/to/your/workspace/models/beingvl/base/`
- Model with extended vocabulary for VQ and vBPE tokens
- Verification logs confirming successful model loading

## Part 2: 3-Stage Training Pipeline

Being-VL employs a 3-stage training methodology that combines curriculum-based data composition with progressive parameter unfreezing.

### Stage 1: Embedding Alignment

```bash
# Edit beingvl/scripts/train-stage-1.sh with your paths:
MODEL_PATH="/path/to/your/workspace/models/beingvl/base"
DATA_PATH="/path/to/your/workspace/data/tokenized/pt/pretrain_data_vbpe.jsonl"
OUTPUT_DIR="/path/to/your/workspace/models/beingvl/stage-1"
LOG_DIR="/path/to/your/workspace/logs/stage-1"

# Run Stage 1 training
bash beingvl/scripts/train-stage-1.sh <master_node_id>
```

### Stage 2: Selective Fine-tuning

```bash
# Edit beingvl/scripts/train-stage-2.sh with your paths:
DATA_PATH="/path/to/your/workspace/data/tokenized/sft_stage2/sft_stage2_data_vbpe.jsonl"
OUTPUT_DIR="/path/to/your/workspace/models/beingvl/stage-2"
LOG_DIR="/path/to/your/workspace/logs/stage-2"

# Run Stage 2 training (auto-detects Stage 1 output)
bash beingvl/scripts/train-stage-2.sh <master_node_id>
```

### Stage 3: Full Fine-tuning

```bash
# Edit beingvl/scripts/train-stage-3.sh with your paths:
DATA_PATH="/path/to/your/workspace/data/tokenized/sft_stage3/sft_stage3_data_vbpe.jsonl"
OUTPUT_DIR="/path/to/your/workspace/models/beingvl/stage-3"
LOG_DIR="/path/to/your/workspace/logs/stage-3"

# Run Stage 3 training (auto-detects Stage 2 output)
bash beingvl/scripts/train-stage-3.sh <master_node_id>
```
