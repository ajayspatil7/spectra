"""
Phase 1 Main Experiment: Query Norm → Attention Entropy Correlation
====================================================================

This script runs the complete Phase 1 experiment:
1. Load model and prepare long-context input
2. Profile all attention layers
3. Compute query norms and attention entropy
4. Calculate correlations per (layer, head)
5. Run randomization control
6. Save results

Usage:
    python scripts/run_experiment.py --context-length 4096
    python scripts/run_experiment.py --context-length 2048 --skip-control
"""

import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ExperimentConfig
from src.data_loader import load_long_context, load_from_shards
from src.hooks import AttentionProfiler
from src.metrics import (
    compute_query_norm,
    compute_attention_entropy,
    compute_correlations,
    run_randomization_control,
    results_to_dataframe,
    results_to_heatmap_matrix,
    print_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Query Norm → Entropy Correlation")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Model name or path")
    parser.add_argument("--context-length", type=int, default=4096,
                        help="Context length in tokens")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Number of samples to process")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing preprocessed shards")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--skip-control", action="store_true",
                        help="Skip randomization control (faster)")
    parser.add_argument("--n-permutations", type=int, default=100,
                        help="Number of permutations for randomization control")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def print_gpu_info():
    """Print GPU information."""
    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)


def load_model(model_name: str):
    """Load model and tokenizer."""
    print(f"\nLoading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}, "
          f"Heads: {model.config.num_attention_heads}")
    
    return model, tokenizer


def run_experiment(args):
    """Run the complete Phase 1 experiment."""
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "=" * 60)
    print("PHASE 1 EXPERIMENT: Query Norm → Attention Entropy")
    print("=" * 60)
    print(f"Timestamp: {timestamp}")
    print(f"Context length: {args.context_length}")
    print(f"Number of samples: {args.n_samples}")
    print(f"Output directory: {output_dir}")
    
    # Check GPU
    print_gpu_info()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    
    # Load model
    start_time = time.time()
    model, tokenizer = load_model(args.model)
    load_time = time.time() - start_time
    print(f"Model load time: {load_time:.1f}s")
    
    # Prepare input data
    print(f"\nPreparing input data...")
    
    # Try to load from shards if n_samples > 1
    if args.n_samples > 1:
        samples = load_from_shards(
            data_dir=args.data_dir,
            n_samples=args.n_samples,
            device="cuda"
        )
        if samples is None:
            print("Falling back to single sample mode")
            samples = [load_long_context(tokenizer, target_length=args.context_length)]
    else:
        samples = [load_long_context(tokenizer, target_length=args.context_length)]
    
    print(f"Processing {len(samples)} samples...")
    
    # Aggregate results across all samples
    all_results = []
    profile_times = []
    
    # Profile attention for each sample
    for sample_idx, inputs in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {sample_idx + 1}/{len(samples)}")
        print(f"{'='*60}")
        
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]
        print(f"Sequence length: {seq_len}")
        
        # Profile attention
        profiler = AttentionProfiler(model)
        
        profile_start = time.time()
        layer_data = profiler.profile(input_ids, compute_attn_probs=True)
        profile_time = time.time() - profile_start
        profile_times.append(profile_time)
        print(f"Profiling time: {profile_time:.1f}s")
        
        # Get metrics
        q_norms = profiler.get_all_query_norms()
        entropy = profiler.get_all_attention_entropy()
        
        # Compute correlations for this sample
        sample_results = compute_correlations(q_norms, entropy, verbose=False)
        all_results.append(sample_results)
        
        # Clear CUDA cache to prevent OOM
        torch.cuda.empty_cache()
    
    # Aggregate correlations across samples
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS ACROSS SAMPLES")
    print(f"{'='*60}")
    
    # Combine all results
    from collections import defaultdict
    aggregated = defaultdict(list)
    
    for sample_results in all_results:
        for result in sample_results:
            key = (result.layer, result.head)
            aggregated[key].append({
                'pearson_r': result.pearson_r,
                'pearson_p': result.pearson_p,
                'spearman_r': result.spearman_r,
                'spearman_p': result.spearman_p,
            })
    
    # Compute mean correlations per (layer, head)
    from src.metrics import CorrelationResult
    results = []
    
    for (layer, head), sample_corrs in aggregated.items():
        mean_pearson_r = np.mean([s['pearson_r'] for s in sample_corrs])
        mean_pearson_p = np.mean([s['pearson_p'] for s in sample_corrs])
        mean_spearman_r = np.mean([s['spearman_r'] for s in sample_corrs])
        mean_spearman_p = np.mean([s['spearman_p'] for s in sample_corrs])
        
        results.append(CorrelationResult(
            layer=layer,
            head=head,
            pearson_r=mean_pearson_r,
            pearson_p=mean_pearson_p,
            spearman_r=mean_spearman_r,
            spearman_p=mean_spearman_p,
        ))
    
    print(f"Aggregated results from {len(samples)} samples")
    print(f"Mean profiling time: {np.mean(profile_times):.1f}s")
    
    # Print summary
    print_summary(results)
    
    profile_time = np.mean(profile_times)
    correlation_time = 0  # Already computed in aggregation
    
    # Randomization control
    null_dist = None
    if not args.skip_control:
        print("\nRunning randomization control...")
        control_start = time.time()
        null_dist = run_randomization_control(
            q_norms, entropy, 
            n_permutations=args.n_permutations,
            seed=args.seed
        )
        control_time = time.time() - control_start
        print(f"Control experiment time: {control_time:.1f}s")
    
    # Save results
    print("\nSaving results...")
    
    # 1. Correlation DataFrame
    df = results_to_dataframe(results)
    csv_path = output_dir / f"correlations_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # 2. Heatmap matrices
    pearson_matrix = results_to_heatmap_matrix(
        results, model.config.num_hidden_layers, model.config.num_attention_heads, 'pearson_r'
    )
    spearman_matrix = results_to_heatmap_matrix(
        results, model.config.num_hidden_layers, model.config.num_attention_heads, 'spearman_r'
    )
    
    matrices_path = output_dir / f"heatmap_matrices_{timestamp}.npz"
    np.savez(matrices_path, pearson=pearson_matrix, spearman=spearman_matrix)
    print(f"  Saved: {matrices_path}")
    
    # 3. Raw data (for further analysis)
    raw_data = {
        'q_norms': q_norms.cpu().numpy(),
        'entropy': entropy.cpu().numpy(),
        'timestamp': timestamp,
        'config': {
            'model': args.model,
            'context_length': args.context_length,
            'n_layers': model.config.num_hidden_layers,
            'n_heads': model.config.num_attention_heads,
        }
    }
    raw_path = output_dir / f"raw_data_{timestamp}.pkl"
    with open(raw_path, 'wb') as f:
        pickle.dump(raw_data, f)
    print(f"  Saved: {raw_path}")
    
    # 4. Summary JSON
    summary = {
        'timestamp': timestamp,
        'model': args.model,
        'context_length': args.context_length,
        'n_samples': len(samples),
        'n_layers': model.config.num_hidden_layers,
        'n_heads': model.config.num_attention_heads,
        'total_pairs': len(results),
        'significant_pairs': sum(1 for r in results if r.is_significant()),
        'mean_pearson_r': float(df['pearson_r'].mean()),
        'std_pearson_r': float(df['pearson_r'].std()),
        'min_pearson_r': float(df['pearson_r'].min()),
        'max_pearson_r': float(df['pearson_r'].max()),
        'mean_abs_pearson_r': float(df['pearson_r'].abs().mean()),
        'times': {
            'model_load': load_time,
            'profiling': profile_time,
            'correlation': correlation_time,
        }
    }
    
    if null_dist:
        # Convert numpy types to Python types for JSON serialization
        summary['randomization_control'] = {
            k: float(v) if hasattr(v, 'item') else v 
            for k, v in null_dist.items()
        }
        summary['times']['control'] = control_time
    
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")
    
    # Final report
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    
    sig_count = sum(1 for r in results if r.is_significant())
    total = len(results)
    sig_pct = 100 * sig_count / total
    
    print(f"\nKey Results:")
    print(f"  Significant correlations: {sig_count}/{total} ({sig_pct:.1f}%)")
    print(f"  Mean |r|: {df['pearson_r'].abs().mean():.4f}")
    print(f"  Max |r|:  {df['pearson_r'].abs().max():.4f}")
    
    if null_dist:
        print(f"\nRandomization Control:")
        print(f"  Shuffled mean |r|: {null_dist['abs_mean_shuffled_r']:.4f}")
        
        # Check if control passed
        if null_dist['abs_mean_shuffled_r'] < 0.1:
            print("  ✅ Control PASSED (shuffled r → ~0)")
        else:
            print("  ⚠️  Control may have issues")
    
    # Go/No-Go decision
    print("\n" + "-" * 60)
    print("PHASE 1 DECISION")
    print("-" * 60)
    
    # Criteria: |r| >= 0.5 in >= 20% of heads, control passed
    threshold_pct = 20
    control_passed = null_dist is None or null_dist['abs_mean_shuffled_r'] < 0.1
    
    if sig_pct >= threshold_pct and control_passed:
        print(f"✅ GO: {sig_pct:.1f}% of heads show significant correlation (threshold: {threshold_pct}%)")
        decision = "GO"
    elif sig_pct >= 10:
        print(f"⚠️  MARGINAL: {sig_pct:.1f}% of heads show significant correlation")
        print("   Consider examining which layers/heads have strong correlations")
        decision = "MARGINAL"
    else:
        print(f"❌ NO-GO: Only {sig_pct:.1f}% of heads show significant correlation")
        decision = "NO-GO"
    
    summary['decision'] = decision
    
    # Update summary with decision
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    
    return summary


if __name__ == "__main__":
    args = parse_args()
    summary = run_experiment(args)
