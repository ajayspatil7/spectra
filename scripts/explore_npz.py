#!/usr/bin/env python3
"""
NPZ File Explorer for Spectra Phase-1
======================================

Comprehensive investigation of .npz files from experiments and preprocessing.

Usage:
    python scripts/explore_npz.py data/processed/shard_00000.npz
    python scripts/explore_npz.py results/raw_data_*.pkl --pickle
    python scripts/explore_npz.py results/heatmap_matrices_*.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def explore_npz(filepath: str, verbose: bool = True):
    """
    Comprehensively explore a .npz file.
    """
    path = Path(filepath)
    
    if not path.exists():
        print(f"❌ File not found: {filepath}")
        return
    
    print("=" * 70)
    print(f"NPZ EXPLORER: {path.name}")
    print("=" * 70)
    print(f"Path: {path.absolute()}")
    print(f"Size: {path.stat().st_size / 1024:.2f} KB")
    print()
    
    # Load file
    data = np.load(filepath, allow_pickle=True)
    
    # List all arrays
    print("-" * 70)
    print("CONTENTS")
    print("-" * 70)
    
    for i, key in enumerate(data.files):
        arr = data[key]
        print(f"\n[{i+1}] '{key}'")
        print(f"    Type:   {type(arr).__name__}")
        print(f"    Dtype:  {arr.dtype}")
        print(f"    Shape:  {arr.shape}")
        print(f"    Size:   {arr.size:,} elements")
        print(f"    Memory: {arr.nbytes / 1024:.2f} KB")
        
        # Numeric statistics
        if np.issubdtype(arr.dtype, np.number):
            # Handle NaN values
            valid = arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr.flatten()
            
            print(f"\n    Statistics:")
            print(f"      Min:    {valid.min():.6f}" if len(valid) > 0 else "      Min:    N/A")
            print(f"      Max:    {valid.max():.6f}" if len(valid) > 0 else "      Max:    N/A")
            print(f"      Mean:   {valid.mean():.6f}" if len(valid) > 0 else "      Mean:   N/A")
            print(f"      Std:    {valid.std():.6f}" if len(valid) > 0 else "      Std:    N/A")
            print(f"      Median: {np.median(valid):.6f}" if len(valid) > 0 else "      Median: N/A")
            
            # Check for NaN/Inf
            if np.issubdtype(arr.dtype, np.floating):
                n_nan = np.isnan(arr).sum()
                n_inf = np.isinf(arr).sum()
                if n_nan > 0 or n_inf > 0:
                    print(f"\n    ⚠️  NaN count: {n_nan:,}")
                    print(f"    ⚠️  Inf count: {n_inf:,}")
            
            # Percentiles
            if len(valid) > 0:
                print(f"\n    Percentiles:")
                for p in [1, 5, 25, 50, 75, 95, 99]:
                    print(f"      P{p:02d}: {np.percentile(valid, p):.6f}")
            
            # Unique values (for small arrays or integers)
            if arr.size < 10000 or np.issubdtype(arr.dtype, np.integer):
                unique = np.unique(arr[~np.isnan(arr)] if np.issubdtype(arr.dtype, np.floating) else arr)
                if len(unique) <= 20:
                    print(f"\n    Unique values ({len(unique)}): {unique}")
                else:
                    print(f"\n    Unique values: {len(unique):,} distinct")
        
        # Sample values
        if verbose:
            print(f"\n    Sample (first 5 elements):")
            flat = arr.flatten()
            sample = flat[:min(5, len(flat))]
            print(f"      {sample}")
            
            if len(arr.shape) >= 2:
                print(f"\n    First row/slice:")
                if len(arr.shape) == 2:
                    print(f"      {arr[0, :min(10, arr.shape[1])]}")
                elif len(arr.shape) == 3:
                    print(f"      [{0}][0]: {arr[0, 0, :min(10, arr.shape[2])]}")
    
    # Correlation heatmap specific analysis
    if 'pearson' in data.files and 'spearman' in data.files:
        print("\n" + "-" * 70)
        print("CORRELATION HEATMAP ANALYSIS")
        print("-" * 70)
        
        pearson = data['pearson']
        spearman = data['spearman']
        
        # Valid correlations
        valid_pearson = pearson[~np.isnan(pearson)]
        valid_spearman = spearman[~np.isnan(spearman)]
        
        print(f"\nPearson correlations:")
        print(f"  Strong positive (r > 0.5):  {(valid_pearson > 0.5).sum()}")
        print(f"  Moderate pos (0.3 < r < 0.5): {((valid_pearson > 0.3) & (valid_pearson <= 0.5)).sum()}")
        print(f"  Weak (|r| < 0.3):           {(np.abs(valid_pearson) < 0.3).sum()}")
        print(f"  Moderate neg (-0.5 < r < -0.3): {((valid_pearson < -0.3) & (valid_pearson >= -0.5)).sum()}")
        print(f"  Strong negative (r < -0.5): {(valid_pearson < -0.5).sum()}")
        
        # Best heads
        if len(pearson.shape) == 2:
            n_layers, n_heads = pearson.shape
            print(f"\nTop 5 strongest correlations:")
            flat_idx = np.argsort(np.abs(pearson).flatten())[::-1]
            for i in range(min(5, len(flat_idx))):
                idx = flat_idx[i]
                layer = idx // n_heads
                head = idx % n_heads
                r = pearson[layer, head]
                if not np.isnan(r):
                    print(f"  Layer {layer:2d}, Head {head:2d}: r = {r:+.4f}")
    
    # Token ID specific analysis (preprocessed shards)
    if 'input_ids' in data.files:
        print("\n" + "-" * 70)
        print("TOKEN SHARD ANALYSIS")
        print("-" * 70)
        
        input_ids = data['input_ids']
        n_sequences, seq_len = input_ids.shape
        
        print(f"\nSequences: {n_sequences:,}")
        print(f"Sequence length: {seq_len}")
        print(f"Total tokens: {input_ids.size:,}")
        
        # Token distribution
        unique_tokens = np.unique(input_ids)
        print(f"Unique tokens: {len(unique_tokens):,}")
        print(f"Token ID range: {input_ids.min()} - {input_ids.max()}")
        
        # Most common tokens
        from collections import Counter
        flat_tokens = input_ids.flatten().tolist()
        token_counts = Counter(flat_tokens)
        print(f"\nTop 10 most frequent tokens:")
        for token, count in token_counts.most_common(10):
            print(f"  Token {token}: {count:,} ({100*count/len(flat_tokens):.2f}%)")
    
    # Query norm / entropy analysis
    if 'q_norms' in data.files or 'entropy' in data.files:
        print("\n" + "-" * 70)
        print("QUERY NORM / ENTROPY ANALYSIS")
        print("-" * 70)
        
        if 'q_norms' in data.files:
            q_norms = data['q_norms']
            valid_norms = q_norms[~np.isnan(q_norms)]
            print(f"\nQuery Norms [{q_norms.shape}]:")
            print(f"  Mean: {valid_norms.mean():.4f}")
            print(f"  Std:  {valid_norms.std():.4f}")
            print(f"  Range: {valid_norms.min():.4f} - {valid_norms.max():.4f}")
        
        if 'entropy' in data.files:
            entropy = data['entropy']
            valid_entropy = entropy[~np.isnan(entropy)]
            n_nan = np.isnan(entropy).sum()
            print(f"\nAttention Entropy [{entropy.shape}]:")
            print(f"  Mean: {valid_entropy.mean():.4f}")
            print(f"  Std:  {valid_entropy.std():.4f}")
            print(f"  Range: {valid_entropy.min():.4f} - {valid_entropy.max():.4f}")
            print(f"  NaN positions: {n_nan:,} ({100*n_nan/entropy.size:.1f}%)")
    
    data.close()
    print("\n" + "=" * 70)


def explore_pickle(filepath: str):
    """
    Explore a pickle file (raw_data_*.pkl).
    """
    import pickle
    
    path = Path(filepath)
    print("=" * 70)
    print(f"PICKLE EXPLORER: {path.name}")
    print("=" * 70)
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type: {type(data).__name__}")
    
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"\n['{key}']")
            if isinstance(value, np.ndarray):
                print(f"  Type:  ndarray")
                print(f"  Shape: {value.shape}")
                print(f"  Dtype: {value.dtype}")
            elif isinstance(value, dict):
                print(f"  Type:  dict with {len(value)} keys")
                print(f"  Keys:  {list(value.keys())}")
            else:
                print(f"  Type:  {type(value).__name__}")
                print(f"  Value: {value}")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Explore NPZ files")
    parser.add_argument("file", type=str, help="Path to .npz or .pkl file")
    parser.add_argument("--pickle", "-p", action="store_true", help="File is a pickle")
    parser.add_argument("--quiet", "-q", action="store_true", help="Less verbose output")
    args = parser.parse_args()
    
    if args.pickle or args.file.endswith('.pkl'):
        explore_pickle(args.file)
    else:
        explore_npz(args.file, verbose=not args.quiet)


if __name__ == "__main__":
    main()
