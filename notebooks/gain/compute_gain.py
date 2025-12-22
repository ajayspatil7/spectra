#!/usr/bin/env python3
"""
Gain Analysis for Phase-1 Experiment-1
======================================

Quantifies how strongly each attention head responds to query magnitude scaling.

Purpose:
- Compute Gain = slope of (mean_max_attn vs log(scale))
- Compare target vs control heads
- Identify if target heads have higher sensitivity

Inputs:
- Intervention CSVs from run_intervention.py

Outputs (saved to notebooks/gain/):
- gain_summary.csv
- gain_plot_*.png

Usage:
    python notebooks/gain/compute_gain.py \
        --target-csv results/intervention/q_scale_intervention_L12_H0_target_*.csv \
        --control-csv results/intervention/q_scale_intervention_L12_H15_control_*.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import glob


# Output directory - all results go here
OUTPUT_DIR = Path("notebooks/gain")


@dataclass
class GainResult:
    """Result of gain computation for a single head."""
    layer: int
    head: int
    head_type: str
    gain_max_attn: float      # Slope of max_attn vs log(scale)
    r_squared: float          # R² of the fit
    std_error: float          # Standard error of the slope
    gain_entropy: Optional[float] = None
    gain_k_eff: Optional[float] = None
    is_monotonic: bool = True


def check_monotonicity(values: np.ndarray, direction: str = "increasing") -> bool:
    """
    Check if values are monotonically increasing or decreasing.
    
    Args:
        values: Array of values
        direction: "increasing" or "decreasing"
        
    Returns:
        True if monotonic in the specified direction
    """
    if direction == "increasing":
        return all(values[i] <= values[i+1] for i in range(len(values)-1))
    else:
        return all(values[i] >= values[i+1] for i in range(len(values)-1))


def compute_linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit linear regression y = m*x + c.
    
    Args:
        x: Independent variable (log of scales)
        y: Dependent variable (metric values)
        
    Returns:
        (slope, intercept, r_squared, std_error)
    """
    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    
    return slope, intercept, r_squared, std_err


def compute_gain_for_head(df: pd.DataFrame) -> GainResult:
    """
    Compute Gain for a single head from intervention CSV data.
    
    Args:
        df: DataFrame with columns [scale, mean_max_attn, layer, head, head_type]
        
    Returns:
        GainResult with computed gain values
    """
    # Sort by scale
    df = df.sort_values('scale').reset_index(drop=True)
    
    # Extract data
    scales = df['scale'].values
    max_attn = df['mean_max_attn'].values
    
    # Get metadata
    layer = int(df['layer'].iloc[0])
    head = int(df['head'].iloc[0])
    head_type = df['head_type'].iloc[0]
    
    # Check monotonicity (max_attn should increase with scale)
    is_monotonic = check_monotonicity(max_attn, direction="increasing")
    
    if not is_monotonic:
        print(f"  ⚠️ Warning: Layer {layer} Head {head} is not monotonic!")
    
    # Compute log of scales
    log_scales = np.log(scales)
    
    # Fit linear regression: max_attn = gain * log(scale) + intercept
    gain, intercept, r_squared, std_error = compute_linear_regression(log_scales, max_attn)
    
    # Compute optional gains for entropy and k_eff
    gain_entropy = None
    gain_k_eff = None
    
    if 'mean_entropy' in df.columns:
        entropy = df['mean_entropy'].values
        gain_entropy, _, _, _ = compute_linear_regression(log_scales, entropy)
    
    if 'mean_k_eff' in df.columns:
        k_eff = df['mean_k_eff'].values
        gain_k_eff, _, _, _ = compute_linear_regression(log_scales, k_eff)
    
    return GainResult(
        layer=layer,
        head=head,
        head_type=head_type,
        gain_max_attn=gain,
        r_squared=r_squared,
        std_error=std_error,
        gain_entropy=gain_entropy,
        gain_k_eff=gain_k_eff,
        is_monotonic=is_monotonic
    )


def plot_gain_fit(
    df: pd.DataFrame, 
    gain_result: GainResult,
    output_path: Path,
    metric: str = "mean_max_attn"
):
    """
    Plot the linear fit of metric vs log(scale).
    
    Args:
        df: DataFrame with intervention data
        gain_result: Computed gain result
        output_path: Where to save the plot
        metric: Which metric to plot
    """
    df = df.sort_values('scale').reset_index(drop=True)
    
    scales = df['scale'].values
    values = df[metric].values
    log_scales = np.log(scales)
    
    # Compute fitted line
    gain = gain_result.gain_max_attn
    intercept = np.mean(values) - gain * np.mean(log_scales)
    fitted = gain * log_scales + intercept
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data points
    ax.scatter(log_scales, values, s=100, c='#3498db', zorder=3, label='Observed')
    
    # Fitted line
    ax.plot(log_scales, fitted, 'r--', linewidth=2, label=f'Fit (Gain={gain:.4f})')
    
    # Labels
    ax.set_xlabel('log(Scale)', fontsize=12)
    ax.set_ylabel('Max Attention Weight', fontsize=12)
    ax.set_title(
        f"Gain Analysis: Layer {gain_result.layer}, Head {gain_result.head} ({gain_result.head_type})\n"
        f"Gain = {gain:.4f}, R² = {gain_result.r_squared:.4f}",
        fontsize=12
    )
    
    # Add scale values as secondary x-axis labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(log_scales)
    ax2.set_xticklabels([f'{s:.2f}' for s in scales])
    ax2.set_xlabel('Scale Factor', fontsize=10)
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def plot_gain_comparison(
    results: List[GainResult],
    output_path: Path
):
    """
    Plot bar chart comparing Gain across heads.
    
    Args:
        results: List of GainResult for different heads
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    labels = [f"L{r.layer}H{r.head}\n({r.head_type})" for r in results]
    gains = [r.gain_max_attn for r in results]
    colors = ['#2ecc71' if r.head_type == 'target' else '#e74c3c' for r in results]
    
    # Bar plot
    bars = ax.bar(labels, gains, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, gain in zip(bars, gains):
        height = bar.get_height()
        ax.annotate(f'{gain:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Head', fontsize=12)
    ax.set_ylabel('Gain (slope of max_attn vs log(scale))', fontsize=12)
    ax.set_title('Gain Comparison Across Heads', fontsize=14, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Target'),
        Patch(facecolor='#e74c3c', label='Control')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def save_gain_summary(results: List[GainResult], output_path: Path):
    """
    Save gain summary to CSV.
    
    Args:
        results: List of GainResult
        output_path: Where to save the CSV
    """
    data = []
    for r in results:
        data.append({
            'layer': r.layer,
            'head': r.head,
            'head_type': r.head_type,
            'gain_max_attn': r.gain_max_attn,
            'r_squared': r.r_squared,
            'std_error': r.std_error,
            'gain_entropy': r.gain_entropy,
            'gain_k_eff': r.gain_k_eff,
            'is_monotonic': r.is_monotonic
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    return df


def find_intervention_csvs(results_dir: str = "results/intervention") -> Dict[str, List[Path]]:
    """
    Find all intervention CSVs in results directory.
    
    Returns:
        Dict with 'target' and 'control' lists of paths
    """
    results_path = Path(results_dir)
    
    csvs = {
        'target': [],
        'control': []
    }
    
    for csv_file in results_path.glob("q_scale_intervention_L*_H*_*.csv"):
        if 'target' in csv_file.name:
            csvs['target'].append(csv_file)
        elif 'control' in csv_file.name:
            csvs['control'].append(csv_file)
    
    return csvs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Gain from intervention experiment results"
    )
    parser.add_argument("--results-dir", type=str, default="results/intervention",
                        help="Directory containing intervention CSVs")
    parser.add_argument("--target-csv", type=str, default=None,
                        help="Explicit path to target head CSV (glob pattern allowed)")
    parser.add_argument("--control-csv", type=str, default=None,
                        help="Explicit path to control head CSV (glob pattern allowed)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("GAIN ANALYSIS — Path A")
    print("=" * 70)
    print("Quantifying head-specific sensitivity to query scaling")
    print("=" * 70)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Find or load CSVs
    csv_files = []
    
    if args.target_csv:
        target_files = glob.glob(args.target_csv)
        csv_files.extend([(f, 'target') for f in target_files])
    
    if args.control_csv:
        control_files = glob.glob(args.control_csv)
        csv_files.extend([(f, 'control') for f in control_files])
    
    # If no explicit CSVs provided, auto-discover
    if not csv_files:
        print("\nAuto-discovering intervention CSVs...")
        found = find_intervention_csvs(args.results_dir)
        csv_files = [(str(f), 'target') for f in found['target']]
        csv_files += [(str(f), 'control') for f in found['control']]
    
    if not csv_files:
        print("❌ No intervention CSVs found!")
        print(f"   Looking in: {args.results_dir}")
        print("   Run intervention experiments first.")
        return
    
    print(f"\nFound {len(csv_files)} CSV files:")
    for f, t in csv_files:
        print(f"  [{t}] {f}")
    
    # Process each CSV
    print("\n" + "=" * 70)
    print("COMPUTING GAIN")
    print("=" * 70)
    
    results = []
    
    for csv_path, expected_type in csv_files:
        print(f"\nProcessing: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required = ['scale', 'mean_max_attn', 'layer', 'head', 'head_type']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  ❌ Missing columns: {missing}")
            continue
        
        # Compute gain
        gain_result = compute_gain_for_head(df)
        results.append(gain_result)
        
        print(f"  Layer: {gain_result.layer}, Head: {gain_result.head}")
        print(f"  Gain (max_attn): {gain_result.gain_max_attn:.4f}")
        print(f"  R²: {gain_result.r_squared:.4f}")
        print(f"  Monotonic: {'✅' if gain_result.is_monotonic else '❌'}")
        
        # Plot individual head
        plot_path = OUTPUT_DIR / f"gain_fit_L{gain_result.layer}_H{gain_result.head}_{gain_result.head_type}.png"
        plot_gain_fit(df, gain_result, plot_path)
    
    if not results:
        print("\n❌ No valid results to process!")
        return
    
    # Save summary
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    summary_path = OUTPUT_DIR / "gain_summary.csv"
    summary_df = save_gain_summary(results, summary_path)
    
    # Plot comparison if multiple heads
    if len(results) > 1:
        comparison_path = OUTPUT_DIR / "gain_comparison.png"
        plot_gain_comparison(results, comparison_path)
    
    # Print summary table
    print("\n" + "=" * 70)
    print("GAIN SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    
    # Compute gain ratio if we have target and control
    targets = [r for r in results if r.head_type == 'target']
    controls = [r for r in results if r.head_type == 'control']
    
    if targets and controls:
        print("\n" + "=" * 70)
        print("TARGET vs CONTROL COMPARISON")
        print("=" * 70)
        
        avg_target_gain = np.mean([r.gain_max_attn for r in targets])
        avg_control_gain = np.mean([r.gain_max_attn for r in controls])
        
        print(f"\n  Average Target Gain: {avg_target_gain:.4f}")
        print(f"  Average Control Gain: {avg_control_gain:.4f}")
        
        if avg_control_gain != 0:
            gain_ratio = avg_target_gain / avg_control_gain
            print(f"  Gain Ratio (Target/Control): {gain_ratio:.2f}x")
            
            if abs(gain_ratio - 1.0) < 0.1:
                print("\n  → Gain ratio ≈ 1: No functional distinction detected")
                print("    Both heads respond similarly to Q scaling")
            elif gain_ratio > 1.5:
                print(f"\n  → Target heads show {gain_ratio:.1f}x higher sensitivity")
                print("    Evidence of head-specific exploitation of Q magnitude")
            else:
                print("\n  → Minor difference in sensitivity")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    for f in OUTPUT_DIR.glob("*"):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
