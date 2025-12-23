#!/usr/bin/env python3
"""
Intervention Results Visualization
==================================

Generates publication-quality plots from intervention experiment results.

Produces three plots:
1. Entropy vs Scale (expected: monotonic decreasing)
2. Max Attention vs Scale (expected: monotonic increasing)
3. k_eff vs Scale (expected: monotonic decreasing)

Usage:
    python scripts/plot_intervention.py --input results/intervention/q_scale_intervention_results.csv
    python scripts/plot_intervention.py --help
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate intervention experiment plots"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to intervention results CSV")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: same as input)")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf", "svg"],
                        help="Output format (default: png)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Resolution for PNG output (default: 150)")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively")
    return parser.parse_args()


def setup_style():
    """Set up matplotlib style for publication-quality plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_metric_vs_scale(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    expected_trend: str,
    output_path: Path,
    format: str = "png",
    dpi: int = 150
):
    """
    Generate a single metric vs scale plot.
    
    Args:
        df: DataFrame with intervention results
        metric: Column name for y-axis (e.g., 'mean_entropy')
        ylabel: Y-axis label
        title: Plot title
        expected_trend: "increasing" or "decreasing"
        output_path: Where to save the plot
        format: Output format
        dpi: Resolution
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get data
    scales = df['scale'].values
    values = df[metric].values
    
    # Check monotonicity
    if expected_trend == "decreasing":
        is_monotonic = all(values[i] >= values[i+1] for i in range(len(values)-1))
        trend_match = values[-1] < values[0]
    else:  # increasing
        is_monotonic = all(values[i] <= values[i+1] for i in range(len(values)-1))
        trend_match = values[-1] > values[0]
    
    # Plot line
    color = '#2ecc71' if trend_match else '#e74c3c'
    ax.plot(scales, values, 'o-', 
            color=color, 
            linewidth=2.5, 
            markersize=10,
            markeredgecolor='white',
            markeredgewidth=2)
    
    # Add error bars if std is available
    std_col = metric.replace('mean_', 'std_')
    if std_col in df.columns:
        stds = df[std_col].values
        ax.fill_between(scales, values - stds, values + stds, 
                        alpha=0.2, color=color)
    
    # Add trend annotation
    trend_text = "↓ " if expected_trend == "decreasing" else "↑ "
    trend_text += f"{expected_trend.upper()}"
    status = "✓ Matches expected" if trend_match else "✗ Does not match"
    mono_status = "Monotonic" if is_monotonic else "Non-monotonic"
    
    # Add annotation box
    annotation = f"Expected: {trend_text}\n{status}\n{mono_status}"
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    ax.annotate(annotation, xy=(0.02, 0.98), xycoords='axes fraction',
                verticalalignment='top', fontsize=10,
                bbox=props)
    
    # Labels and title
    ax.set_xlabel('Scale Factor (s)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add layer/head info if available
    if 'layer' in df.columns and 'head' in df.columns:
        layer = df['layer'].iloc[0]
        head = df['head'].iloc[0]
        head_type = df.get('head_type', ['target']).iloc[0]
        ax.set_title(f"{title}\nLayer {layer}, Head {head} ({head_type})", 
                    fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks
    ax.set_xticks(scales)
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    
    return fig


def plot_combined(
    df: pd.DataFrame,
    output_path: Path,
    format: str = "png",
    dpi: int = 150
):
    """
    Generate combined plot with all three metrics.
    
    Creates a 1x3 subplot figure for compact visualization.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    scales = df['scale'].values
    
    metrics = [
        ('mean_entropy', 'Entropy (H)', 'decreasing', '#3498db'),
        ('mean_max_attn', 'Max Attention (α_max)', 'increasing', '#e74c3c'),
        ('mean_k_eff', 'Effective Span (k_eff)', 'decreasing', '#9b59b6'),
    ]
    
    for ax, (metric, ylabel, expected, color) in zip(axes, metrics):
        values = df[metric].values
        
        # Check trend
        if expected == "decreasing":
            trend_match = values[-1] < values[0]
        else:
            trend_match = values[-1] > values[0]
        
        # Plot
        line_color = color if trend_match else '#95a5a6'
        ax.plot(scales, values, 'o-', 
                color=line_color, 
                linewidth=2.5, 
                markersize=8,
                markeredgecolor='white',
                markeredgewidth=1.5)
        
        # Add std shading
        std_col = metric.replace('mean_', 'std_')
        if std_col in df.columns:
            stds = df[std_col].values
            ax.fill_between(scales, values - stds, values + stds, 
                            alpha=0.2, color=line_color)
        
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel(ylabel)
        
        # Add trend indicator
        status = "✓" if trend_match else "✗"
        trend_arrow = "↓" if expected == "decreasing" else "↑"
        ax.set_title(f"{ylabel}\n{status} Expected: {trend_arrow}", fontsize=11)
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks(scales)
    
    # Overall title
    if 'layer' in df.columns and 'head' in df.columns:
        layer = df['layer'].iloc[0]
        head = df['head'].iloc[0]
        head_type = df.get('head_type', ['target']).iloc[0]
        fig.suptitle(f"Causal Intervention Results: Layer {layer}, Head {head} ({head_type})",
                    fontsize=14, fontweight='bold', y=1.02)
    else:
        fig.suptitle("Causal Intervention Results", 
                    fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    
    return fig


def main():
    args = parse_args()
    
    # Setup style
    setup_style()
    
    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"\n--- Loading Results ---")
    print(f"  Input: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get layer/head for filename
    layer = df.get('layer', [0]).iloc[0] if 'layer' in df.columns else 0
    head = df.get('head', [0]).iloc[0] if 'head' in df.columns else 0
    suffix = f"_L{layer}_H{head}" if 'layer' in df.columns else ""
    
    print(f"\n--- Generating Plots ---")
    
    # Plot 1: Entropy vs Scale
    plot_metric_vs_scale(
        df,
        metric='mean_entropy',
        ylabel='Attention Entropy (H)',
        title='Entropy vs Query Scale',
        expected_trend='decreasing',
        output_path=output_dir / f"entropy_vs_scale{suffix}.{args.format}",
        format=args.format,
        dpi=args.dpi
    )
    
    # Plot 2: Max Attention vs Scale
    plot_metric_vs_scale(
        df,
        metric='mean_max_attn',
        ylabel='Maximum Attention Weight (α_max)',
        title='Max Attention vs Query Scale',
        expected_trend='increasing',
        output_path=output_dir / f"max_attn_vs_scale{suffix}.{args.format}",
        format=args.format,
        dpi=args.dpi
    )
    
    # Plot 3: k_eff vs Scale
    plot_metric_vs_scale(
        df,
        metric='mean_k_eff',
        ylabel='Effective Attention Span (k_eff)',
        title='Effective Span vs Query Scale',
        expected_trend='decreasing',
        output_path=output_dir / f"keff_vs_scale{suffix}.{args.format}",
        format=args.format,
        dpi=args.dpi
    )
    
    # Combined plot
    plot_combined(
        df,
        output_path=output_dir / f"intervention_combined{suffix}.{args.format}",
        format=args.format,
        dpi=args.dpi
    )
    
    print(f"\n--- Complete ---")
    print(f"  Plots saved to: {output_dir}")
    
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
