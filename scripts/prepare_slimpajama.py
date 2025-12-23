#!/usr/bin/env python3
"""
SlimPajama Dataset Preparation
===============================

Downloads SlimPajama-6B and generates context-length-specific samples.

Steps:
1. Download DKYoon/SlimPajama-6B to data/slimpajama
2. Generate 64 random samples at each context length
3. Save as NPZ (for training) and JSON (human-readable)

Usage:
    python scripts/prepare_slimpajama.py
    
Outputs:
    data/slimpajama/          - Raw dataset cache
    data/ctx128/              - 128-token samples
    data/ctx512/              - 512-token samples
    data/ctx1024/             - 1024-token samples
    data/ctx2048/             - 2048-token samples
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset


def download_slimpajama(cache_dir: Path):
    """Download SlimPajama-6B dataset."""
    print("\n--- Downloading SlimPajama-6B ---")
    print(f"  Cache dir: {cache_dir}")
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Try different loading approaches
    try:
        # Method 1: Direct load with explicit config
        dataset = load_dataset(
            "DKYoon/SlimPajama-6B",
            split="train",
            cache_dir=str(cache_dir),
        )
        print(f"  Dataset size: {len(dataset)} samples")
        return dataset
    except ValueError as e:
        print(f"  Direct load failed: {e}")
        print("  Trying alternative method...")
    
    try:
        # Method 2: Load with streaming first, then convert
        print("  Loading with streaming mode...")
        dataset = load_dataset(
            "DKYoon/SlimPajama-6B",
            split="train",
            streaming=True,
        )
        
        # Take first N samples and convert to list
        print("  Converting streamed samples to list...")
        samples = []
        for i, sample in enumerate(dataset):
            samples.append(sample)
            if i >= 10000:  # Limit to 10k samples
                break
            if i % 1000 == 0:
                print(f"    Loaded {i} samples...")
        
        print(f"  Loaded {len(samples)} samples")
        return samples
    except Exception as e2:
        print(f"  Streaming also failed: {e2}")
        raise RuntimeError(f"Could not load SlimPajama-6B: {e2}")


def generate_samples_from_dataset(
    dataset,
    tokenizer,
    target_length: int,
    output_dir: Path,
    n_samples: int = 64,
    seed: int = 42
):
    """
    Generate samples at a specific context length from cached dataset.
    
    Args:
        dataset: HuggingFace dataset
        tokenizer: Tokenizer
        target_length: Target token count
        output_dir: Output directory
        n_samples: Number of samples
        seed: Random seed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Generating {n_samples} samples of {target_length} tokens ---")
    print(f"  Output: {output_dir}")
    
    np.random.seed(seed)
    
    # Sample random indices
    dataset_size = len(dataset)
    random_indices = np.random.choice(dataset_size, size=min(n_samples * 10, dataset_size), replace=False)
    
    samples = []
    sample_texts = []  # For JSON
    
    for idx in random_indices:
        if len(samples) >= n_samples:
            break
        
        text = dataset[int(idx)]["text"]
        if not text:
            continue
        
        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        # Skip if too short
        if len(tokens) < target_length:
            continue
        
        # Take first target_length tokens
        tokens = tokens[:target_length]
        
        samples.append(tokens)
        
        # Decode for JSON (human-readable)
        decoded_text = tokenizer.decode(tokens)
        sample_texts.append({
            "sample_idx": len(samples),
            "source_idx": int(idx),
            "n_tokens": len(tokens),
            "text_preview": decoded_text[:500] + "..." if len(decoded_text) > 500 else decoded_text,
            "full_text": decoded_text
        })
        
        print(f"    Sample {len(samples)}/{n_samples}: {len(tokens)} tokens (source idx: {idx})")
    
    if not samples:
        print(f"  Warning: No samples generated!")
        return 0
    
    # Save as NPZ
    all_tokens = np.array(samples, dtype=np.int32)
    shard_path = output_dir / "shard_000.npz"
    np.savez_compressed(
        shard_path,
        input_ids=all_tokens,
        n_samples=len(samples),
        target_length=target_length
    )
    print(f"  Saved NPZ: {shard_path}")
    print(f"  Shape: {all_tokens.shape}")
    
    # Save as JSON (human-readable)
    json_path = output_dir / "samples.json"
    with open(json_path, 'w') as f:
        json.dump({
            "metadata": {
                "target_length": target_length,
                "n_samples": len(samples),
                "seed": seed,
                "generated_at": datetime.now().isoformat()
            },
            "samples": sample_texts
        }, f, indent=2)
    print(f"  Saved JSON: {json_path}")
    
    return len(samples)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SlimPajama samples")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Tokenizer model")
    parser.add_argument("--lengths", type=str, default="128,512,1024,2048",
                        help="Comma-separated context lengths")
    parser.add_argument("--n-samples", type=int, default=64,
                        help="Number of samples per length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default="data/slimpajama",
                        help="Where to cache SlimPajama dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("SLIMPAJAMA DATASET PREPARATION")
    print("=" * 60)
    
    cache_dir = Path(args.cache_dir)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Download/load SlimPajama
    dataset = download_slimpajama(cache_dir)
    
    # Parse lengths
    lengths = [int(x.strip()) for x in args.lengths.split(",")]
    print(f"\nContext lengths: {lengths}")
    print(f"Samples per length: {args.n_samples}")
    
    # Generate for each length
    results = {}
    for length in lengths:
        output_dir = Path(f"data/ctx{length}")
        count = generate_samples_from_dataset(
            dataset,
            tokenizer,
            target_length=length,
            output_dir=output_dir,
            n_samples=args.n_samples,
            seed=args.seed
        )
        results[length] = count
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  SlimPajama cached: {cache_dir}")
    for length, count in results.items():
        print(f"  ctx{length}: {count} samples → data/ctx{length}/")
        print(f"         NPZ: data/ctx{length}/shard_000.npz")
        print(f"        JSON: data/ctx{length}/samples.json")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
