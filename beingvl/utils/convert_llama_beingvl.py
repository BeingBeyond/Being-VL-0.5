#!/usr/bin/env python3
"""
Convert LLaMA model to Being-VL base model for training.

This script initializes a base weight for Being-VL training by:
1. Copying LLaMA model weights and Being tokenizer configuration
2. Resizing token embeddings to accommodate extended vocabulary
3. Saving the initialized model

Usage:
    python convert_llama_to_being.py --llama_path /path/to/llama --output_path /path/to/output
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor, ChameleonVQVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LLaMA model to Being-VL base model")
    
    parser.add_argument(
        "--llama_path",
        type=str,
        required=True,
        help="Path to the original LLaMA model directory"
    )
    
    parser.add_argument(
        "--being_tokenizer_config_path", 
        type=str,
        required=True,
        help="Path to Being tokenizer configuration directory"
    )
    
    parser.add_argument(
        "--being_vq_path",
        type=str, 
        required=True,
        help="Path to pre-trained Being VQ-VAE model"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the converted Being-VL base model"
    )
    
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=128256,
        help="Base vocabulary size (default: 128256 for LLaMA 3.1)"
    )
    
    parser.add_argument(
        "--extended_vocab_size",
        type=int,
        default=16389,  # 5 + 8192 + 8192
        help="Extended vocabulary size for image tokens (default: 16389)"
    )
    
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="PyTorch dtype for model loading (default: bfloat16)"
    )
    
    parser.add_argument(
        "--verify_loading",
        action="store_true",
        help="Verify the converted model can be loaded with BeingVL"
    )
    
    return parser.parse_args()


def copy_model_files(llama_path: str, being_tokenizer_config_path: str, output_path: str):
    """Copy LLaMA weights and Being tokenizer configuration to output directory."""
    print(f"Creating output directory: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    # Copy safetensors files from LLaMA
    print(f"Copying LLaMA weights from: {llama_path}")
    llama_files = Path(llama_path).glob("*safetensors*")
    for file_path in llama_files:
        shutil.copy2(file_path, output_path)
        print(f"  Copied: {file_path.name}")
    
    # Copy all files from Being tokenizer config
    print(f"Copying Being tokenizer config from: {being_tokenizer_config_path}")
    tokenizer_files = Path(being_tokenizer_config_path).glob("*")
    for file_path in tokenizer_files:
        if file_path.is_file():
            shutil.copy2(file_path, output_path)
            print(f"  Copied: {file_path.name}")


def update_config(output_path: str, vocab_size: int):
    """Update model configuration with correct vocabulary size."""
    config_path = os.path.join(output_path, "config.json")
    
    print(f"Updating config.json with vocab_size: {vocab_size}")
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    config['vocab_size'] = vocab_size
    
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=2)
    
    print(f"✓ Updated vocab_size to {vocab_size}")


def load_and_convert_model(
    output_path: str, 
    being_vq_path: str, 
    vocab_size: int, 
    extended_vocab_size: int,
    torch_dtype: str
):
    """Load model, add VQ-VAE components, and resize embeddings."""
    
    # Convert string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16, 
        "float32": torch.float32
    }
    torch_dtype_obj = dtype_map[torch_dtype]
    
    print(f"Loading base model from: {output_path}")
    model = ChameleonForConditionalGeneration.from_pretrained(
        output_path,
        torch_dtype=torch_dtype_obj,
    )
    
    print(f"Loading VQGAN from: {being_vq_path}")
    model.model.vqmodel = ChameleonVQVAE.from_pretrained(
        being_vq_path,
        torch_dtype=model.dtype,
    )
    
    print(f"Loading processor from: {output_path}")
    processor = ChameleonProcessor.from_pretrained(output_path)
    
    # Calculate total vocabulary size
    total_vocab_size = vocab_size + extended_vocab_size
    print(f"Resizing token embeddings to: {total_vocab_size} ({vocab_size} + {extended_vocab_size})")
    
    model.resize_token_embeddings(total_vocab_size)
    
    return model, processor


def save_converted_model(model, processor, output_path: str):
    """Save the converted model and processor."""
    print(f"Saving converted model to: {output_path}")
    model.save_pretrained(output_path)
    
    print(f"Saving processor to: {output_path}")
    processor_files = processor.save_pretrained(output_path)
    print(f"  Saved processor files: {processor_files}")


def verify_with_beingvl(output_path: str, torch_dtype: str):
    """Verify the converted model can be loaded with BeingVL."""
    try:
        from beingvl import BeingVL
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        torch_dtype_obj = dtype_map[torch_dtype]
        
        print("Verifying model loading with BeingVL...")
        model = BeingVL(model_path=output_path)
        model.load_model(torch_dtype=torch_dtype_obj)
        
        print("✓ Model successfully loaded with BeingVL")
        
    except ImportError:
        print("⚠ BeingVL not available for verification (import error)")
    except Exception as e:
        print(f"✗ Error loading with BeingVL: {e}")
        raise


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Converting LLaMA to Being-VL Base Model")
    print("=" * 60)
    print(f"LLaMA path: {args.llama_path}")
    print(f"Being tokenizer config: {args.being_tokenizer_config_path}")
    print(f"Being VQ path: {args.being_vq_path}")
    print(f"Output path: {args.output_path}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Extended vocab size: {args.extended_vocab_size}")
    print(f"PyTorch dtype: {args.torch_dtype}")
    print()
    
    try:
        # Step 1: Copy model files and configuration
        copy_model_files(
            args.llama_path, 
            args.being_tokenizer_config_path, 
            args.output_path
        )
        
        # Step 2: Update configuration
        update_config(args.output_path, args.vocab_size)
        
        # Step 3: Load and convert model
        model, processor = load_and_convert_model(
            args.output_path,
            args.being_vq_path,
            args.vocab_size,
            args.extended_vocab_size,
            args.torch_dtype
        )
        
        # Step 4: Save converted model
        save_converted_model(model, processor, args.output_path)
        
        # Step 5: Optional verification
        if args.verify_loading:
            verify_with_beingvl(args.output_path, args.torch_dtype)
        
        print()
        print("=" * 60)
        print("✓ Conversion completed successfully!")
        print(f"✓ Being-VL base model saved to: {args.output_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during conversion: {e}")
        raise


if __name__ == "__main__":
    main()