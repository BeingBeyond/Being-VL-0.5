# Data Preparation Guide

This document covers the complete data preparation pipeline for Being-VL training, including VQ token extraction, vBPE training, and dataset tokenization.

## Prerequisites

Before starting, ensure you have:
- Set up the workspace directory structure (see README.md)
- Installed Being-VL and dependencies
- Downloaded the required pre-trained models:
  - LLaMA 3.1-8B model
  - Being VQ-GAN model (8K vocabulary)
  - Being tokenizer config files
- Downloaded raw image datasets for visual BPE tokenizer training
- Downloaded caption/conversation datasets for PT/SFT training
- *Due to data licensing, we cannot provide direct downloads. You need to obtain datasets by yourself. Details of the data components can be found in our paper.*

## Part 1: vBPE Training Data Preparation

The first step is preparing VQ tokens for training the visual BPE tokenizer.

### Step 1.1: VQ Encoding Images to NPY

Convert your image dataset to VQ tokens using the pre-trained VQ-GAN model. This creates a compressed representation of images as discrete tokens.

```python
import torch
import numpy as np
from PIL import Image
from transformers import ChameleonProcessor, ChameleonVQVAE

# Load processor and VQ-GAN model (using consistent workspace paths)
processor = ChameleonProcessor.from_pretrained("/path/to/your/workspace/models/being-tokenizer")
vqgan = ChameleonVQVAE.from_pretrained(
    "/path/to/your/workspace/models/being-vq-8k",
    torch_dtype=torch.bfloat16,
)
vqgan.eval()

# Prepare your image list from workspace
images_list = [
    "/path/to/your/workspace/data/images/image1.jpg",
    "/path/to/your/workspace/data/images/image2.jpg",
    "/path/to/your/workspace/data/images/image3.jpg",
]

# Load and process images
images = [Image.open(image_path).convert('RGB') for image_path in images_list]
images_pixels = processor.image_processor(
    images, 
    return_tensors='pt'
)["pixel_values"].to(vqgan.device, vqgan.dtype)

# Encode images to tokens
_, _, toks = vqgan.encode(images_pixels)

# Save tokens to workspace vq_tokens directory
toks = toks.cpu().numpy()  # Shape: (num_images, 32, 32)
np.save("/path/to/your/workspace/data/vq_tokens/train_tokens.npy", toks)

print(f"Saved {toks.shape[0]} image tokens with shape {toks.shape}")
```

### Step 1.2: Train vBPE Tokenizer

Once you have VQ tokens, train the visual BPE tokenizer to create an extended vocabulary for efficient image representation.

```bash
python beingvl/train/train_vbpe.py \
    --data_path /path/to/your/workspace/data/vq_tokens/train_tokens.npy \
    --output_path /path/to/your/workspace/data/vbpe/vbpe.pkl \
    --num_merges 8192 \
    --init_size 8192 \
    --vocab_pad 128256 \
    --vocab_end 136448 \
    --validate_data
```

**Training Parameters:**
- `--num_merges`: Number of BPE merge operations (default: 8192)
- `--init_size`: Initial vocabulary size (8192 for VQ tokens)
- `--vocab_pad`: Padding to end of original text vocabulary (128256 for LLaMA 3)
- `--vocab_end`: End position for VQ tokens before BPE tokens start (136448)
- `--validate_data`: Validate input data format before training
- `--save_intermediate`: Save intermediate training statistics

**Expected Output:**
- `vbpe.pkl` file containing the trained visual BPE tokenizer
- Training logs showing merge operations and compression statistics

## Part 2: Training Data Preparation

After vBPE training, prepare your datasets for the 3-stage training pipeline.

### Step 2.1: Data Format Requirements

Being-VL accepts data in LLaVA format. See [LLaVA Data](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) for format details.

**Input Format Example:**
```json
{
  "image": "coco/train2017/000000175217.jpg", 
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat is the color of the sink and vanity in the bathroom?"
    },
    {
      "from": "gpt", 
      "value": "The sink and vanity in the bathroom are beige in color."
    }
  ]
}
```

**Output Format (after tokenization):**
```json
{
  "image": "coco/train2017/000000175217.jpg",
  "conversations": [...],
  "token_0": [128000, 128260],
  "token_1": [128258, 3923, 374, ...],
  "token_image_new": [131257, 130547, 135495, ...]
}
```

### Step 2.2: Tokenize Datasets

Convert your prepared datasets into tokenized format for training:

```bash
# For pretraining datasets
python beingvl/utils/tokenize_dataset.py \
    --mode pt \
    --json_path /path/to/your/workspace/data/annotations/pretrain_data.json \
    --image_path /path/to/your/workspace/data/images \
    --output_dir /path/to/your/workspace/data/tokenized/pt \
    --model_path /path/to/your/workspace/models/beingvl/base \
    --vbpe_path /path/to/your/workspace/data/vbpe/vbpe.pkl \
    --dataset_name "pretrain_data" \
    --batch_size 32 \
    --num_workers 32 \
    --torch_dtype bfloat16

# For fine-tuning datasets (Stage 2)
python beingvl/utils/tokenize_dataset.py \
    --mode sft \
    --json_path /path/to/your/workspace/data/annotations/sft_stage2_data.json \
    --image_path /path/to/your/workspace/data/images \
    --output_dir /path/to/your/workspace/data/tokenized/sft_stage2 \
    --model_path /path/to/your/workspace/models/beingvl/base \
    --vbpe_path /path/to/your/workspace/data/vbpe/vbpe.pkl \
    --dataset_name "sft_stage2_data" \
    --batch_size 32 \
    --num_workers 32 \
    --torch_dtype bfloat16

# For fine-tuning datasets (Stage 3)
python beingvl/utils/tokenize_dataset.py \
    --mode sft \
    --json_path /path/to/your/workspace/data/annotations/sft_stage3_data.json \
    --image_path /path/to/your/workspace/data/images \
    --output_dir /path/to/your/workspace/data/tokenized/sft_stage3 \
    --model_path /path/to/your/workspace/models/beingvl/base \
    --vbpe_path /path/to/your/workspace/data/vbpe/vbpe.pkl \
    --dataset_name "sft_stage3_data" \
    --batch_size 32 \
    --num_workers 32 \
    --torch_dtype bfloat16
```

**Key parameters:**
- `--mode`: Set to `pt` for pretraining or `sft` for fine-tuning
- `--model_path`: Use the initialized beingvl-base model (from Step 1 model initialization)
- `--vbpe_path`: Path to the trained vBPE tokenizer (from vBPE training step)
- `--dataset_name`: Custom name for the dataset (will be used in output filenames)

**Output files:**
- `*_base.jsonl`: Base tokenization without vBPE compression
- `*_vbpe.jsonl`: Enhanced tokenization with vBPE compression (use this for training)
