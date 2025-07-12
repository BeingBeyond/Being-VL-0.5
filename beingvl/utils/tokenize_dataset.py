#!/usr/bin/env python3
"""
Tokenize datasets for Being-VL training (both pretraining and fine-tuning).

This script processes image-text datasets and converts them into tokenized format
for Being-VL training. It supports both base tokenization and extended vocabulary
(vBPE) tokenization.

Usage:
    # Pretraining dataset
    python tokenize_dataset.py --mode pt --json_path /path/to/data.json --image_path /path/to/images
    
    # Fine-tuning dataset
    python tokenize_dataset.py --mode sft --json_path /path/to/data.json --image_path /path/to/images
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from beingvl import BeingVL


# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ImageDataset(Dataset):
    """Dataset for loading images and associated data."""
    
    def __init__(self, json_data: List[Dict], image_path: str):
        # Filter data to only include entries with images
        self.json_data = [item for item in json_data if "image" in item]
        self.image_path = image_path
        print(f"Dataset initialized with {len(self.json_data)} image-text pairs")

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        dt = self.json_data[idx]
        image_file = os.path.join(self.image_path, dt["image"])
        
        try:
            img = Image.open(image_file).convert('RGB')
            return img, dt
        except Exception as e:
            print(f"Warning: Failed to load image {image_file}: {e}")
            # Return a placeholder black image
            img = Image.new('RGB', (224, 224), color='black')
            return img, dt


def collate_fn(batch):
    """Collate function for DataLoader."""
    imgs, data = zip(*batch)
    return list(imgs), list(data)


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize datasets for Being-VL training")
    
    # Required arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pt", "sft"],
        help="Training mode: 'pt' for pretraining, 'sft' for supervised fine-tuning"
    )
    
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the dataset"
    )
    
    parser.add_argument(
        "--image_path", 
        type=str,
        required=True,
        help="Path to the directory containing images"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for tokenized datasets"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the Being-VL model"
    )
    
    parser.add_argument(
        "--vbpe_path",
        type=str,
        help="Path to the vBPE tokenizer (.pkl file). If provided, will generate vBPE tokenized version"
    )
    
    # Model parameters
    parser.add_argument(
        "--vbpe_init_size",
        type=int,
        default=8192,
        help="vBPE initial vocabulary size (default: 8192)"
    )
    
    parser.add_argument(
        "--vbpe_vocab_pad",
        type=int,
        default=128261,  # 128256 + 5
        help="vBPE vocabulary padding (default: 128261)"
    )
    
    parser.add_argument(
        "--vbpe_vocab_end",
        type=int,
        default=136453,  # 128256 + 5 + 8192
        help="vBPE vocabulary end (default: 136453)"
    )
    
    # Processing parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of worker processes for data loading"
    )
    
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="PyTorch dtype for model loading (default: bfloat16)"
    )
    
    # Output naming
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name for output files (auto-generated if not provided)"
    )
    
    return parser.parse_args()


def load_dataset(json_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file."""
    print(f"Loading dataset from: {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"✓ Dataset loaded successfully: {len(data)} entries")
        return data
    except Exception as e:
        raise ValueError(f"Failed to load dataset from {json_path}: {e}")


def initialize_model(args) -> BeingVL:
    """Initialize the Being-VL model."""
    print(f"Initializing Being-VL model from: {args.model_path}")
    
    # Convert string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map[args.torch_dtype]
    
    # Initialize model
    model = BeingVL(
        model_path=args.model_path,
        vbpe_path=args.vbpe_path if args.vbpe_path else None,
    )
    
    # Load model with specified parameters
    model.device_map = "auto"
    model.load_model(
        torch_dtype=torch_dtype,
        vbpe_init_size=args.vbpe_init_size,
        vbpe_vocab_pad=args.vbpe_vocab_pad,
        vbpe_vocab_end=args.vbpe_vocab_end,
    )
    
    print(f"✓ Model loaded successfully")
    if args.vbpe_path:
        print(f"✓ vBPE tokenizer loaded from: {args.vbpe_path}")
    
    return model


def generate_output_paths(args) -> tuple:
    """Generate output file paths."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate dataset name if not provided
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        json_file = Path(args.json_path).stem
        dataset_name = f"{args.mode}_{json_file}"
    
    # Generate output paths
    base_output = os.path.join(args.output_dir, f"{dataset_name}_base.jsonl")
    vbpe_output = os.path.join(args.output_dir, f"{dataset_name}_vbpe.jsonl") if args.vbpe_path else None
    
    return base_output, vbpe_output


def select_tokenization_method(mode: str):
    """Select appropriate tokenization method based on mode."""
    if mode == "pt":
        # Pretraining: usually single-turn conversations
        return "batch_tokenize_single_turn"
    elif mode == "sft":
        # Fine-tuning: can be multi-turn conversations
        return "batch_tokenize_multi_turn_multi_entry"
    else:
        raise ValueError(f"Unknown mode: {mode}")


def process_dataset(model: BeingVL, dataset: ImageDataset, args) -> None:
    """Process the dataset and generate tokenized output."""
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Select tokenization method
    tokenization_method = select_tokenization_method(args.mode)
    tokenize_func = getattr(model, tokenization_method)
    
    print(f"Processing dataset with method: {tokenization_method}")
    print(f"Total batches to process: {len(dataloader)}")
    
    # Clear any existing tokenization results
    model.tokenize_result = []
    
    # Process batches
    with tqdm(total=len(dataloader), desc="Processing batches") as pbar:
        for batch_idx, (images, data) in enumerate(dataloader):
            try:
                tokenize_func(images, data)
                pbar.update(1)
            except Exception as e:
                print(f"Warning: Error processing batch {batch_idx}: {e}")
                pbar.update(1)
                continue
    
    print(f"✓ Dataset processing completed")
    print(f"  Processed entries: {len(model.tokenize_result)}")


def save_results(model: BeingVL, base_output: str, vbpe_output: str = None):
    """Save tokenization results."""
    print(f"Saving base tokenization to: {base_output}")
    model.save_tokenize_result(base_output)
    print(f"✓ Base tokenization saved: {base_output}")
    
    if vbpe_output and model.vbpe_path:
        print(f"Generating vBPE tokenization...")
        
        # Copy tokenization results for vBPE processing
        tokenize_result = model.tokenize_result.copy()
        
        # Add vBPE token transformations
        model.vbpe.parallel_add_trans_token(tokenize_result)
        
        print(f"Saving vBPE tokenization to: {vbpe_output}")
        model.vbpe.save_tokenize_result(vbpe_output)
        print(f"✓ vBPE tokenization saved: {vbpe_output}")


def main():
    args = parse_args()
    
    print("=" * 60)
    print(f"Tokenizing Dataset for Being-VL ({args.mode.upper()} mode)")
    print("=" * 60)
    print(f"JSON path: {args.json_path}")
    print(f"Image path: {args.image_path}")
    print(f"Model path: {args.model_path}")
    print(f"vBPE path: {args.vbpe_path or 'None (base tokenization only)'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    try:
        # Step 1: Load dataset
        json_data = load_dataset(args.json_path)
        print()
        
        # Step 2: Initialize model
        model = initialize_model(args)
        print()
        
        # Step 3: Load data into model
        model.json_data = json_data
        
        # Step 4: Create dataset and generate output paths
        dataset = ImageDataset(json_data, args.image_path)
        base_output, vbpe_output = generate_output_paths(args)
        
        print(f"Output files:")
        print(f"  Base tokenization: {base_output}")
        if vbpe_output:
            print(f"  vBPE tokenization: {vbpe_output}")
        print()
        
        # Step 5: Process dataset
        process_dataset(model, dataset, args)
        print()
        
        # Step 6: Save results
        save_results(model, base_output, vbpe_output)
        print()
        
        print()
        print("=" * 60)
        print("✓ Dataset tokenization completed successfully!")
        print(f"✓ Base tokenization: {base_output}")
        if vbpe_output:
            print(f"✓ vBPE tokenization: {vbpe_output}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during tokenization: {e}")
        raise


if __name__ == "__main__":
    main()