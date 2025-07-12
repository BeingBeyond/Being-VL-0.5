#!/bin/bash

# Script for converting LLaMA to Being-VL base model
# Modify the paths below according to your setup

# Configuration
LLAMA_PATH="/path/to/llama-3.1-8b"
BEING_TOKENIZER_CONFIG_PATH="/path/to/being-tokenizer-config"
BEING_VQ_PATH="/path/to/being-vq-8k"
OUTPUT_PATH="/output_path/to/beingvl-vq8k-bpe8k"

# Run conversion
python beingvl/utils/convert_llama_beingvl.py \
    --llama_path "$LLAMA_PATH" \
    --being_tokenizer_config_path "$BEING_TOKENIZER_CONFIG_PATH" \
    --being_vq_path "$BEING_VQ_PATH" \
    --output_path "$OUTPUT_PATH" \
    --vocab_size 128256 \
    --extended_vocab_size 16389 \
    --torch_dtype bfloat16 \
    --verify_loading
