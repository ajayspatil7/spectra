#!/usr/bin/env python3
"""
Generate Long Context Samples
=============================

Creates preprocessed samples at specific context lengths.
Stores each length in a separate directory.

Usage:
    python scripts/generate_context_samples.py
    
Outputs:
    data/ctx512/   - 512 token samples
    data/ctx1024/  - 1024 token samples  
    data/ctx2048/  - 2048 token samples
"""

import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset


def generate_samples(
    tokenizer,
    target_length: int,
    output_dir: Path,
    n_samples: int = 5,
    seed: int = 42
):
    """
    Generate samples at specific context length.
    
    Args:
        tokenizer: HuggingFace tokenizer
        target_length: Target number of tokens
        output_dir: Where to save
        n_samples: Number of samples to generate
        seed: Random seed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Generating {n_samples} samples of {target_length} tokens ---")
    print(f"  Output: {output_dir}")
    
    # Use C4 (streamable) or allenai/c4
    print("  Loading C4 dataset (streamable)...")
    try:
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"  Trying alternative dataset...")
        dataset = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True, trust_remote_code=True)
    
    samples = []
    sample_idx = 0
    
    np.random.seed(seed)
    skip_count = np.random.randint(0, 1000)  # Random starting point
    
    iterator = iter(dataset)
    for _ in range(skip_count):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(dataset)
    
    attempts = 0
    max_attempts = 10000
    
    while len(samples) < n_samples and attempts < max_attempts:
        attempts += 1
        try:
            article = next(iterator)
        except StopIteration:
            break
        
        text = article.get("text", "")
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
        sample_idx += 1
        print(f"    Sample {sample_idx}/{n_samples}: {len(tokens)} tokens")
    
    # Save as numpy shard
    if samples:
        all_tokens = np.array(samples, dtype=np.int32)
        
        shard_path = output_dir / f"shard_000.npz"
        np.savez_compressed(
            shard_path,
            input_ids=all_tokens,
            n_samples=len(samples),
            target_length=target_length
        )
        print(f"  Saved: {shard_path}")
        print(f"  Shape: {all_tokens.shape}")
    else:
        print(f"  Warning: No samples generated!")
    
    return len(samples)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate context samples")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Tokenizer model")
    parser.add_argument("--lengths", type=str, default="512,1024,2048",
                        help="Comma-separated context lengths")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of samples per length")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("GENERATE CONTEXT SAMPLES")
    print("=" * 60)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Parse lengths
    lengths = [int(x.strip()) for x in args.lengths.split(",")]
    print(f"Context lengths: {lengths}")
    print(f"Samples per length: {args.n_samples}")
    
    # Generate for each length
    results = {}
    for length in lengths:
        output_dir = Path(f"data/ctx{length}")
        count = generate_samples(
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
    for length, count in results.items():
        print(f"  ctx{length}: {count} samples → data/ctx{length}/")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
