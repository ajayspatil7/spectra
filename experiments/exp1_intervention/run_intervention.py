#!/usr/bin/env python3
"""
Causal Intervention Experiment Script
======================================

Runs Phase-1 Experiment-1: Causal intervention on query norm.

This script:
1. Loads a single sample (512 tokens)
2. Targets a specific (layer, head) pair
3. Scales Q vector by factors [0.5, 0.75, 1.0, 1.25, 1.5]
4. Measures attention metrics: entropy, max_attn, k_eff
5. Saves results to CSV for analysis

Expected outcome: Monotonic relationships between scale and metrics.

Usage:
    python scripts/run_intervention.py --target-layer 12 --target-head 0
    python scripts/run_intervention.py --help

Remote execution (SageMaker):
    python scripts/run_intervention.py \
        --target-layer 12 \
        --target-head 0 \
        --output-dir results/intervention
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_loader import load_long_context, load_from_shards
from src.intervention import (
    InterventionProfiler, 
    InterventionConfig,
    verify_intervention_isolation,
    run_baseline_sanity_check
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Causal Intervention Experiment on Query Norm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (layer 12, head 0)
  python scripts/run_intervention.py

  # Specify target layer and head
  python scripts/run_intervention.py --target-layer 10 --target-head 5

  # Custom scale factors
  python scripts/run_intervention.py --scales "0.25,0.5,1.0,2.0,4.0"

  # Run control experiment on a different head
  python scripts/run_intervention.py --target-layer 12 --target-head 15 --head-type control
        """
    )
    
    # Target configuration
    parser.add_argument("--target-layer", type=int, default=12,
                        help="Layer index to intervene on (default: 12)")
    parser.add_argument("--target-head", type=int, default=0,
                        help="Head index to intervene on (default: 0)")
    parser.add_argument("--head-type", type=str, default="target",
                        choices=["target", "control"],
                        help="Type of head: 'target' (high correlation) or 'control' (low correlation)")
    
    # Intervention settings
    parser.add_argument("--scales", type=str, default="0.5,0.75,1.0,1.25,1.5",
                        help="Comma-separated scale factors (default: 0.5,0.75,1.0,1.25,1.5)")
    
    # Data settings
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Which sample to use for intervention (default: 0)")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing preprocessed shards")
    parser.add_argument("--context-length", type=int, default=512,
                        help="Context length in tokens (default: 512)")
    
    # Model settings
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B",
                        help="Model name or path")
    
    # Output settings
    parser.add_argument("--output-dir", type=str, default="results/intervention",
                        help="Output directory for results")
    
    # Experiment settings
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--verify-isolation", action="store_true",
                        help="Run intervention isolation verification test")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print detailed progress")
    
    return parser.parse_args()


def print_experiment_header(args, timestamp: str):
    """Print experiment configuration header."""
    print("\n" + "=" * 70)
    print("PHASE-1 EXPERIMENT-1: CAUSAL INTERVENTION ON QUERY NORM")
    print("=" * 70)
    print(f"\nTimestamp: {timestamp}")
    print(f"\n--- Target Configuration ---")
    print(f"  Layer: {args.target_layer}")
    print(f"  Head: {args.target_head}")
    print(f"  Head Type: {args.head_type}")
    print(f"\n--- Intervention Settings ---")
    print(f"  Scales: {args.scales}")
    print(f"  Context Length: {args.context_length}")
    print(f"\n--- Output ---")
    print(f"  Directory: {args.output_dir}")
    print("=" * 70)


def print_gpu_info():
    """Print GPU information."""
    print("\n--- GPU Configuration ---")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute: {props.major}.{props.minor}")


def load_model(model_name: str):
    """Load model and tokenizer."""
    print(f"\n--- Loading Model ---")
    print(f"  Model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Heads: {model.config.num_attention_heads}")
    print(f"  Hidden Size: {model.config.hidden_size}")
    
    return model, tokenizer


def load_single_sample(args, tokenizer) -> torch.Tensor:
    """Load a single sample for intervention."""
    print(f"\n--- Loading Sample ---")
    print(f"  Sample Index: {args.sample_idx}")
    print(f"  Data Directory: {args.data_dir}")
    
    # Try to load from shards
    samples = load_from_shards(
        data_dir=args.data_dir,
        n_samples=args.sample_idx + 1,
        device="cuda"
    )
    
    if samples is not None and len(samples) > args.sample_idx:
        sample = samples[args.sample_idx]
        input_ids = sample["input_ids"]
        print(f"  Source: Preprocessed shards")
    else:
        # Fallback to generated sample
        print(f"  Falling back to generated sample")
        sample = load_long_context(tokenizer, target_length=args.context_length)
        input_ids = sample["input_ids"]
    
    print(f"  Token Length: {input_ids.shape[1]}")
    
    return input_ids


def run_experiment(args):
    """Run the causal intervention experiment."""
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print header
    print_experiment_header(args, timestamp)
    print_gpu_info()
    
    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    
    # Load model
    start_time = time.time()
    model, tokenizer = load_model(args.model)
    load_time = time.time() - start_time
    print(f"  Load Time: {load_time:.1f}s")
    
    # Load sample
    input_ids = load_single_sample(args, tokenizer)
    
    # Parse scales
    scales = [float(s.strip()) for s in args.scales.split(",")]
    print(f"\n--- Intervention Scales ---")
    print(f"  {scales}")
    
    # Create intervention config
    config = InterventionConfig(
        target_layer=args.target_layer,
        target_head=args.target_head,
        scales=scales,
        context_length=args.context_length,
    )
    
    # Verify intervention isolation (optional)
    if args.verify_isolation:
        print("\n" + "=" * 70)
        print("INTERVENTION ISOLATION VERIFICATION")
        print("=" * 70)
        passed = verify_intervention_isolation(
            model, input_ids,
            target_layer=args.target_layer,
            target_head=args.target_head,
            scale=2.0
        )
        if not passed:
            print("\n⚠️ WARNING: Intervention isolation test failed!")
            print("Results may be unreliable.")
    
    # 🔴 ADDITION 1: Baseline sanity check
    print("\n" + "=" * 70)
    print("BASELINE SANITY CHECK")
    print("=" * 70)
    sanity_passed, sanity_comparison = run_baseline_sanity_check(
        model, input_ids,
        target_layer=args.target_layer,
        target_head=args.target_head
    )
    if not sanity_passed:
        print("\n⚠️ CRITICAL WARNING: Baseline sanity check failed!")
        print("Hook may be introducing artifacts. Results are unreliable.")
        print("Proceeding anyway for diagnostic purposes...")
    
    # 🔴 ADDITION 2: Explicit directional expectations
    print("\n" + "=" * 70)
    print("DIRECTIONAL EXPECTATIONS (HYPOTHESIS)")
    print("=" * 70)
    print("\nIf query magnitude is a causal control signal, then:")
    print("")
    print("  As scale ↑ (0.5 → 1.5):")
    print("    • entropy  → ↓ DECREASING  (sharper attention)")
    print("    • max_attn → ↑ INCREASING  (more confident)")
    print("    • k_eff    → ↓ DECREASING  (fewer keys needed)")
    print("")
    print("Failure to observe these trends suggests:")
    print("  → Phase-0 correlation may be epiphenomenal")
    print("  → This head may not use Q norm as control signal")
    print("=" * 70)
    
    # Create profiler and run sweep
    print("\n" + "=" * 70)
    print("RUNNING INTERVENTION SWEEP")
    print("=" * 70)
    
    profiler = InterventionProfiler(
        model,
        target_layer=args.target_layer,
        target_head=args.target_head,
        config=config
    )
    
    sweep_start = time.time()
    results = profiler.run_intervention_sweep(
        input_ids,
        scales=scales,
        verbose=args.verbose
    )
    sweep_time = time.time() - sweep_start
    
    # Convert to DataFrame
    df = InterventionProfiler.results_to_dataframe(
        results,
        target_layer=args.target_layer,
        target_head=args.target_head,
        head_type=args.head_type
    )
    
    # Save results
    csv_filename = f"q_scale_intervention_L{args.target_layer}_H{args.target_head}_{args.head_type}_{timestamp}.csv"
    csv_path = output_dir / csv_filename
    df.to_csv(csv_path, index=False)
    
    # Also save canonical filename for easy access
    canonical_path = output_dir / "q_scale_intervention_results.csv"
    df.to_csv(canonical_path, index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\n--- Results Summary ---")
    print(f"\nScale  | Entropy  | Max Attn | k_eff")
    print("-" * 45)
    for _, row in df.iterrows():
        print(f"{row['scale']:5.2f}  | {row['mean_entropy']:7.4f}  | {row['mean_max_attn']:7.4f}  | {row['mean_k_eff']:6.2f}")
    
    print(f"\n--- Trend Analysis ---")
    # Check for expected trends
    entropies = df['mean_entropy'].values
    max_attns = df['mean_max_attn'].values
    k_effs = df['mean_k_eff'].values
    
    # Simple trend check (first vs last)
    entropy_trend = "↓ DECREASING" if entropies[-1] < entropies[0] else "↑ INCREASING"
    max_attn_trend = "↑ INCREASING" if max_attns[-1] > max_attns[0] else "↓ DECREASING"
    keff_trend = "↓ DECREASING" if k_effs[-1] < k_effs[0] else "↑ INCREASING"
    
    print(f"  Entropy: {entropy_trend} (expected: ↓ DECREASING)")
    print(f"  Max Attn: {max_attn_trend} (expected: ↑ INCREASING)")
    print(f"  k_eff: {keff_trend} (expected: ↓ DECREASING)")
    
    # Check monotonicity
    entropy_monotonic = all(entropies[i] >= entropies[i+1] for i in range(len(entropies)-1))
    max_attn_monotonic = all(max_attns[i] <= max_attns[i+1] for i in range(len(max_attns)-1))
    keff_monotonic = all(k_effs[i] >= k_effs[i+1] for i in range(len(k_effs)-1))
    
    print(f"\n--- Monotonicity Check ---")
    print(f"  Entropy: {'✅ Monotonic' if entropy_monotonic else '❌ Non-monotonic'}")
    print(f"  Max Attn: {'✅ Monotonic' if max_attn_monotonic else '❌ Non-monotonic'}")
    print(f"  k_eff: {'✅ Monotonic' if keff_monotonic else '❌ Non-monotonic'}")
    
    print(f"\n--- Output Files ---")
    print(f"  Results CSV: {csv_path}")
    print(f"  Canonical CSV: {canonical_path}")
    print(f"  Sweep Time: {sweep_time:.1f}s")
    
    # Interpretation guidance
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    if entropy_monotonic and max_attn_monotonic:
        print("\n✅ Results support the hypothesis:")
        print("   Query magnitude functions as a CAUSAL CONTROL SIGNAL")
        print("   regulating attention confidence and sharpness.")
    elif args.head_type == "control":
        print("\n📊 Control head results:")
        print("   Non-monotonic trends are expected for control heads.")
        print("   This validates target-specificity of the mechanism.")
    else:
        print("\n⚠️ Results are inconclusive:")
        print("   Consider trying different target heads or layers.")
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("  1. Run: python scripts/plot_intervention.py --input", canonical_path)
    print("  2. Compare with control head results")
    print("=" * 70)
    
    return df, csv_path


if __name__ == "__main__":
    args = parse_args()
    df, csv_path = run_experiment(args)
