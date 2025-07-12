#!/usr/bin/env python3
"""
Train Visual BPE Tokenizer for Being-VL.

This script trains a Visual BPE (vBPE) tokenizer from image tokens extracted by VQ-GAN.
The trained tokenizer compresses image token sequences using byte-pair encoding with
adjacency matrix-based merging for improved efficiency.

Usage:
    python train_vbpe.py --data_path /path/to/tokens.npy --output_path /path/to/vbpe.pkl
"""

import argparse
import os
from pathlib import Path

import numpy as np
from beingvl import vBPE


def parse_args():
    parser = argparse.ArgumentParser(description="Train Visual BPE Tokenizer")
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the .npy file containing image tokens (shape: num_images, 32, 32)"
    )
    
    parser.add_argument(
        "--output_path", 
        type=str,
        required=True,
        help="Output path for the trained vBPE vocabulary (.pkl file)"
    )
    
    parser.add_argument(
        "--num_merges",
        type=int,
        default=8192,
        help="Number of BPE merge operations to perform (default: 8192)"
    )
    
    parser.add_argument(
        "--init_size",
        type=int,
        default=8192,
        help="Initial vocabulary size (default: 8192)"
    )
    
    parser.add_argument(
        "--vocab_pad",
        type=int,
        default=128261,
        help="Vocabulary padding offset. VQ tokens will be added after (text tokens + special tokens) = (128256 + 5) by default"
    )
    
    parser.add_argument(
        "--vocab_end",
        type=int,
        default=136453,
        help="Vocabulary (text + special + vq) end index. BPE tokens will be added after this"
    )
    
    parser.add_argument(
        "--validate_data",
        action="store_true",
        help="Validate input data format and statistics"
    )
    
    parser.add_argument(
        "--save_intermediate",
        action="store_true",
        help="Save intermediate training statistics"
    )
    
    return parser.parse_args()


def validate_data(data_path: str, vocab_pad: int):
    """Validate the input data format and statistics."""
    print("Validating input data...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        data = np.load(data_path)
    except Exception as e:
        raise ValueError(f"Failed to load data from {data_path}: {e}")
    
    print(f"✓ Data loaded successfully")
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    
    # Check expected shape format
    if len(data.shape) != 3:
        raise ValueError(f"Expected 3D data (num_images, height, width), got shape: {data.shape}")
    
    if data.shape[1] != 32 or data.shape[2] != 32:
        print(f"⚠ Warning: Expected 32x32 image tokens, got {data.shape[1]}x{data.shape[2]}")
    
    # Check data range
    min_val, max_val = data.min(), data.max()
    print(f"  Value range: [{min_val}, {max_val}]")
    
    # Apply vocab_pad offset and check adjusted range
    adjusted_data = data - vocab_pad
    adj_min, adj_max = adjusted_data.min(), adjusted_data.max()
    print(f"  Adjusted range (after -{vocab_pad}): [{adj_min}, {adj_max}]")
    
    if adj_min < 0:
        print(f"⚠ Warning: Adjusted data contains negative values (min: {adj_min})")
    
    print(f"✓ Data validation completed")
    return data


def load_training_data(data_path: str, vocab_pad: int):
    """Load and prepare training data."""
    print(f"Loading training data from: {data_path}")
    
    train_codes = np.load(data_path) - vocab_pad
    
    print(f"Training codes loaded successfully:")
    print(f"  Original shape: {train_codes.shape}")
    print(f"  Min value: {train_codes.min()}")
    print(f"  Max value: {train_codes.max()}")
    print(f"  Data type: {train_codes.dtype}")
    
    return train_codes


def initialize_vbpe(init_size: int, vocab_pad: int, vocab_end: int):
    """Initialize the vBPE tokenizer."""
    print(f"Initializing vBPE tokenizer:")
    print(f"  Initial vocabulary size: {init_size}")
    print(f"  Vocabulary padding: {vocab_pad}")
    print(f"  Vocabulary end: {vocab_end}")
    
    vbpe = vBPE(
        init_size=init_size,
        vocab_pad=vocab_pad,
        vocab_end=vocab_end,
    )
    
    return vbpe


def train_vbpe(vbpe, train_codes, num_merges: int, save_intermediate: bool = False, output_dir: str = None):
    """Train the vBPE tokenizer."""
    print(f"Starting vBPE training with {num_merges} merge operations...")
    
    if save_intermediate and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Intermediate files will be saved to: {output_dir}")
    
    # Train the tokenizer
    vbpe.train(train_codes, num_merges=num_merges)
    
    print(f"✓ vBPE training completed")
    print(f"  Total vocabulary size: {vbpe.total_size}")
    print(f"  Number of merge operations: {len(vbpe.ext_vocab)}")
    
    return vbpe


def save_vbpe(vbpe, output_path: str):
    """Save the trained vBPE tokenizer."""
    print(f"Saving trained vBPE to: {output_path}")
    
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vbpe.save_vocab(output_path)
    print(f"✓ vBPE tokenizer saved successfully")


def verify_saved_model(output_path: str):
    """Verify the saved model can be loaded."""
    try:
        print("Verifying saved model...")
        test_vbpe = vBPE()
        test_vbpe.load_vocab(output_path)
        print(f"✓ Model verification successful")
        print(f"  Loaded vocabulary size: {len(test_vbpe.ext_vocab)}")
        
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        raise


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Training Visual BPE Tokenizer for Being-VL")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print(f"Number of merges: {args.num_merges}")
    print(f"Initial vocab size: {args.init_size}")
    print(f"Vocab padding: {args.vocab_pad}")
    print(f"Vocab end: {args.vocab_end}")
    print()
    
    try:
        # Step 1: Validate data if requested
        if args.validate_data:
            validate_data(args.data_path, args.vocab_pad)
            print()
        
        # Step 2: Load training data
        train_codes = load_training_data(args.data_path, args.vocab_pad)
        print()
        
        # Step 3: Initialize vBPE
        vbpe = initialize_vbpe(args.init_size, args.vocab_pad, args.vocab_end)
        print()
        
        # Step 4: Train vBPE
        output_dir = str(Path(args.output_path).parent) if args.save_intermediate else None
        vbpe = train_vbpe(vbpe, train_codes, args.num_merges, args.save_intermediate, output_dir)
        print()
        
        # Step 5: Save trained model
        save_vbpe(vbpe, args.output_path)
        print()
        
        # Step 6: Verify saved model
        verify_saved_model(args.output_path)
        
        print()
        print("=" * 60)
        print("✓ vBPE training completed successfully!")
        print(f"✓ Trained tokenizer saved to: {args.output_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during vBPE training: {e}")
        raise


if __name__ == "__main__":
    main()