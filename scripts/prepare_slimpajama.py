#!/usr/bin/env python3
"""
SlimPajama Dataset Preparation
===============================

Downloads SlimPajama-6B using huggingface_hub (bypasses fsspec bug)
and generates context-length-specific samples.

Usage:
    python scripts/prepare_slimpajama.py
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, list_repo_files


def download_slimpajama(cache_dir: Path, max_files: int = 5):
    """Download SlimPajama-6B parquet files using huggingface_hub."""
    print("\n--- Downloading SlimPajama-6B ---")
    print(f"  Cache dir: {cache_dir}")
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    repo_id = "DKYoon/SlimPajama-6B"
    
    # List files in repo
    print("  Listing repository files...")
    files = list_repo_files(repo_id=repo_id)
    parquet_files = [f for f in files if f.endswith('.parquet')]
    
    print(f"  Found {len(parquet_files)} parquet files")
    
    # Download first few parquet files
    downloaded = []
    for i, pf in enumerate(parquet_files[:max_files]):
        print(f"  Downloading {i+1}/{min(max_files, len(parquet_files))}: {pf}")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=pf,
            cache_dir=str(cache_dir),
            repo_type="dataset"
        )
        downloaded.append(local_path)
    
    print(f"  Downloaded {len(downloaded)} files")
    return downloaded


def load_samples_from_parquet(parquet_files, max_samples: int = 10000):
    """Load samples from parquet files."""
    import pandas as pd
    
    print(f"\n  Loading samples from {len(parquet_files)} parquet files...")
    
    all_samples = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        texts = df['text'].tolist()
        all_samples.extend(texts)
        print(f"    Loaded {len(texts)} samples from {Path(pf).name}")
        
        if len(all_samples) >= max_samples:
            break
    
    print(f"  Total samples: {len(all_samples)}")
    return all_samples[:max_samples]


def generate_samples(
    texts: list,
    tokenizer,
    target_length: int,
    output_dir: Path,
    n_samples: int = 64,
    seed: int = 42
):
    """Generate samples at a specific context length."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n--- Generating {n_samples} samples of {target_length} tokens ---")
    print(f"  Output: {output_dir}")
    
    np.random.seed(seed)
    
    # Shuffle texts
    indices = np.random.permutation(len(texts))
    
    samples = []
    sample_texts = []
    
    for idx in indices:
        if len(samples) >= n_samples:
            break
        
        text = texts[idx]
        if not text or len(text) < 100:
            continue
        
        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) < target_length:
            continue
        
        tokens = tokens[:target_length]
        samples.append(tokens)
        
        decoded = tokenizer.decode(tokens)
        sample_texts.append({
            "sample_idx": len(samples),
            "source_idx": int(idx),
            "n_tokens": len(tokens),
            "text_preview": decoded[:500] + "..." if len(decoded) > 500 else decoded,
            "full_text": decoded
        })
        
        print(f"    Sample {len(samples)}/{n_samples}: {len(tokens)} tokens")
    
    if not samples:
        print(f"  Warning: No samples generated!")
        return 0
    
    # Save NPZ
    all_tokens = np.array(samples, dtype=np.int32)
    shard_path = output_dir / "shard_000.npz"
    np.savez_compressed(shard_path, input_ids=all_tokens, n_samples=len(samples), target_length=target_length)
    print(f"  Saved NPZ: {shard_path} (shape: {all_tokens.shape})")
    
    # Save JSON
    json_path = output_dir / "samples.json"
    with open(json_path, 'w') as f:
        json.dump({
            "metadata": {"target_length": target_length, "n_samples": len(samples), "seed": seed, "generated_at": datetime.now().isoformat()},
            "samples": sample_texts
        }, f, indent=2)
    print(f"  Saved JSON: {json_path}")
    
    return len(samples)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--lengths", default="128,512,1024,2048")
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data/slimpajama")
    parser.add_argument("--max-files", type=int, default=5, help="Max parquet files to download")
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
    
    # Download parquet files
    parquet_files = download_slimpajama(cache_dir, max_files=args.max_files)
    
    # Load samples
    texts = load_samples_from_parquet(parquet_files)
    
    # Generate for each length
    lengths = [int(x.strip()) for x in args.lengths.split(",")]
    print(f"\nContext lengths: {lengths}")
    
    results = {}
    for length in lengths:
        output_dir = Path(f"data/ctx{length}")
        count = generate_samples(texts, tokenizer, length, output_dir, args.n_samples, args.seed)
        results[length] = count
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for length, count in results.items():
        print(f"  ctx{length}: {count} samples")
    print("\nDone!")


if __name__ == "__main__":
    main()
