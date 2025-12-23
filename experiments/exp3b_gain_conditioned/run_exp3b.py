#!/usr/bin/env python3
"""
EXP3b — Gain-Conditioned Q vs K Scaling
========================================

FINAL EXPERIMENT IN SPECTRA

Scientific question:
    Does Q–K asymmetry in attention entropy emerge selectively in high-gain heads,
    or is attention always governed by generic logit magnitude?

Why this experiment exists:
    - Exp3a showed Q ≈ K symmetry for a medium-gain head
    - Exp2b showed strong heterogeneity of gain across heads
    - Exp3b tests whether control regimes differ by gain class

This is NOT exploratory. This is a CONFIRMATORY, STRATIFIED CONTROL EXPERIMENT.

Heads tested (LOCKED — do not change):
    Head A — High-gain:   Layer 20, Head 19 (gain ≈ 0.804)
    Head B — Medium-gain: Layer 14, Head 8  (gain ≈ 0.45-0.55)
    Head C — Low-gain:    Layer 0,  Head 23 (gain ≈ 0.017)

Usage:
    python experiments/exp3b_gain_conditioned/run_exp3b.py

All outputs saved to: experiments/exp3b_gain_conditioned/
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_loader import load_long_context, load_from_shards
from src.metrics import compute_attention_entropy


# ============================================================
# EXPERIMENTAL CONSTANTS — DO NOT MODIFY
# ============================================================

# Scaling factors (LOCKED)
ALPHAS = [0.5, 0.75, 1.0, 1.25, 1.5]

# Target heads (LOCKED — from EXP2b gain analysis)
TARGET_HEADS = {
    'high_gain': {'layer': 20, 'head': 19, 'expected_gain': 0.804},
    'medium_gain': {'layer': 14, 'head': 8, 'expected_gain': 0.50},
    'low_gain': {'layer': 0, 'head': 23, 'expected_gain': 0.017},
}

# Output directory
OUTPUT_DIR = Path("experiments/exp3b_gain_conditioned")


# ============================================================
# CORE PROFILER CLASS
# ============================================================

class Exp3bProfiler:
    """
    Profiler for EXP3b: Gain-Conditioned Q vs K Scaling.
    
    This is a minimal, focused implementation that:
    - Scales Q or K (never both, never V)
    - Applies scaling only at target layer/head
    - Computes entropy with identical logic to previous experiments
    
    NO modifications to core attention logic. Reproducibility > speed.
    """
    
    def __init__(self, model):
        self.model = model
        self.config = model.config
        
        self.n_layers = self.config.num_hidden_layers
        self.n_heads = self.config.num_attention_heads
        self.n_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.n_heads
        
        self._hidden_states: Dict[int, torch.Tensor] = {}
        self.hooks: List = []
    
    def _get_hidden_hook(self, layer_idx: int):
        """
        Hook to capture pre-attention hidden states.
        
        We capture after input LayerNorm, matching inputs to QKV projection.
        This is the correct intervention point for LLaMA architecture.
        """
        def hook_fn(module, input, output):
            self._hidden_states[layer_idx] = output.detach()
        return hook_fn
    
    def _compute_qkv_with_scaling(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        target_layer: int,
        target_head: int,
        q_scale: float = 1.0,
        k_scale: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q, K with optional scaling.
        
        INVARIANTS:
        - Scaling applied ONLY at target_layer, target_head
        - V is NEVER modified
        - All other heads remain unchanged
        """
        layer = self.model.model.layers[layer_idx]
        attn = layer.self_attn
        
        batch_size, seq_len, _ = hidden_states.shape
        
        with torch.no_grad():
            Q = attn.q_proj(hidden_states)
            K = attn.k_proj(hidden_states)
            
            Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            
            # === SCALING (only at target layer/head) ===
            if layer_idx == target_layer:
                if q_scale != 1.0:
                    Q[:, target_head, :, :] = Q[:, target_head, :, :] * q_scale
                
                if k_scale != 1.0:
                    # Handle GQA: map target_head to corresponding KV head
                    kv_head_idx = target_head % self.n_kv_heads
                    K[:, kv_head_idx, :, :] = K[:, kv_head_idx, :, :] * k_scale
        
        return Q, K
    
    def _compute_attention_probs(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Compute attention probabilities with causal mask.
        
        Uses scaled dot-product attention with √d_k scaling.
        Applies causal mask. No approximations.
        """
        batch_size, n_heads, seq_len, head_dim = Q.shape
        _, n_kv_heads, _, _ = K.shape
        
        # Expand K for GQA
        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            K = K.repeat_interleave(n_rep, dim=1)
        
        # Scaled dot-product attention
        scale = head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        return torch.softmax(scores, dim=-1)
    
    def register_hooks(self):
        """Register hooks on all layers."""
        self.remove_hooks()
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            hook = layer.input_layernorm.register_forward_hook(
                self._get_hidden_hook(layer_idx)
            )
            self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._hidden_states = {}
    
    def run_single_condition(
        self,
        input_ids: torch.Tensor,
        target_layer: int,
        target_head: int,
        alpha: float,
        scaling_type: str  # "Q" or "K"
    ) -> float:
        """
        Run a single experimental condition.
        
        Args:
            input_ids: Input tokens
            target_layer: Layer to intervene
            target_head: Head to intervene
            alpha: Scale factor
            scaling_type: "Q" or "K"
            
        Returns:
            Mean entropy for target head
        """
        hidden_states = self._hidden_states[target_layer]
        
        if scaling_type == "Q":
            Q, K = self._compute_qkv_with_scaling(
                target_layer, hidden_states, target_layer, target_head,
                q_scale=alpha, k_scale=1.0
            )
        else:  # K scaling
            Q, K = self._compute_qkv_with_scaling(
                target_layer, hidden_states, target_layer, target_head,
                q_scale=1.0, k_scale=alpha
            )
        
        attn_probs = self._compute_attention_probs(Q, K)
        
        # Extract target head
        target_attn = attn_probs[:, target_head:target_head+1, :, :]
        
        # Compute entropy (same as all previous experiments)
        entropy = compute_attention_entropy(target_attn, ignore_first_n=2)
        
        entropy_np = entropy.squeeze().cpu().numpy()
        valid = ~np.isnan(entropy_np)
        
        return float(np.mean(entropy_np[valid]))
    
    def run_head_comparison(
        self,
        input_ids: torch.Tensor,
        target_layer: int,
        target_head: int,
        alphas: List[float],
        head_name: str,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run full Q vs K comparison for one head.
        
        Args:
            input_ids: Input tokens
            target_layer: Layer index
            target_head: Head index
            alphas: Scale factors
            head_name: Name for logging
            verbose: Print progress
            
        Returns:
            DataFrame with results
        """
        results = []
        
        if verbose:
            print(f"\n--- {head_name}: Layer {target_layer}, Head {target_head} ---")
        
        # Q-scaling
        if verbose:
            print("  Q-scaling:", end=" ")
        for alpha in alphas:
            entropy = self.run_single_condition(
                input_ids, target_layer, target_head, alpha, "Q"
            )
            results.append({
                'alpha': alpha,
                'scaling_type': 'Q',
                'entropy': entropy,
                'layer': target_layer,
                'head': target_head,
                'head_name': head_name
            })
            if verbose:
                print(f"α={alpha:.2f}→{entropy:.3f}", end=" ")
        if verbose:
            print()
        
        # K-scaling
        if verbose:
            print("  K-scaling:", end=" ")
        for alpha in alphas:
            entropy = self.run_single_condition(
                input_ids, target_layer, target_head, alpha, "K"
            )
            results.append({
                'alpha': alpha,
                'scaling_type': 'K',
                'entropy': entropy,
                'layer': target_layer,
                'head': target_head,
                'head_name': head_name
            })
            if verbose:
                print(f"α={alpha:.2f}→{entropy:.3f}", end=" ")
        if verbose:
            print()
        
        return pd.DataFrame(results)


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_entropy_comparison(df: pd.DataFrame, output_path: Path, head_name: str):
    """
    Plot raw entropy: Q-scaling vs K-scaling.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    q_data = df[df['scaling_type'] == 'Q'].sort_values('alpha')
    k_data = df[df['scaling_type'] == 'K'].sort_values('alpha')
    
    alphas = q_data['alpha'].values
    
    ax.plot(alphas, q_data['entropy'].values, 'o-', 
            color='#3498db', linewidth=2.5, markersize=10,
            markeredgecolor='white', markeredgewidth=2,
            label='Q-scaling')
    
    ax.plot(alphas, k_data['entropy'].values, 's--', 
            color='#e74c3c', linewidth=2.5, markersize=10,
            markeredgecolor='white', markeredgewidth=2,
            label='K-scaling')
    
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    
    layer = df['layer'].iloc[0]
    head = df['head'].iloc[0]
    
    ax.set_xlabel('Scaling Factor (α)', fontsize=12)
    ax.set_ylabel('Attention Entropy (H)', fontsize=12)
    ax.set_title(f'Entropy Response: {head_name}\nLayer {layer}, Head {head}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(alphas)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_delta_entropy(df: pd.DataFrame, output_path: Path, head_name: str):
    """
    Plot ΔH = H(α) - H(α=1.0).
    
    THIS IS THE PRIMARY FIGURE.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    q_data = df[df['scaling_type'] == 'Q'].sort_values('alpha')
    k_data = df[df['scaling_type'] == 'K'].sort_values('alpha')
    
    alphas = q_data['alpha'].values
    
    # Find baseline (α=1.0)
    q_baseline = q_data[q_data['alpha'] == 1.0]['entropy'].values[0]
    k_baseline = k_data[k_data['alpha'] == 1.0]['entropy'].values[0]
    
    # Compute delta
    q_delta = q_data['entropy'].values - q_baseline
    k_delta = k_data['entropy'].values - k_baseline
    
    ax.plot(alphas, q_delta, 'o-', 
            color='#3498db', linewidth=2.5, markersize=10,
            markeredgecolor='white', markeredgewidth=2,
            label='Q-scaling')
    
    ax.plot(alphas, k_delta, 's--', 
            color='#e74c3c', linewidth=2.5, markersize=10,
            markeredgecolor='white', markeredgewidth=2,
            label='K-scaling')
    
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    layer = df['layer'].iloc[0]
    head = df['head'].iloc[0]
    
    ax.set_xlabel('Scaling Factor (α)', fontsize=12)
    ax.set_ylabel('ΔH = H(α) - H(α=1.0)', fontsize=12)
    ax.set_title(f'Delta Entropy: {head_name}\nLayer {layer}, Head {head}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xticks(alphas)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_combined_comparison(all_results: Dict[str, pd.DataFrame], output_path: Path):
    """
    Plot all three heads side by side for direct comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = {'Q': '#3498db', 'K': '#e74c3c'}
    titles = ['High Gain (L20 H19)', 'Medium Gain (L14 H8)', 'Low Gain (L0 H23)']
    
    for idx, (name, df) in enumerate(all_results.items()):
        ax = axes[idx]
        
        q_data = df[df['scaling_type'] == 'Q'].sort_values('alpha')
        k_data = df[df['scaling_type'] == 'K'].sort_values('alpha')
        
        alphas = q_data['alpha'].values
        
        # Compute delta
        q_baseline = q_data[q_data['alpha'] == 1.0]['entropy'].values[0]
        k_baseline = k_data[k_data['alpha'] == 1.0]['entropy'].values[0]
        
        q_delta = q_data['entropy'].values - q_baseline
        k_delta = k_data['entropy'].values - k_baseline
        
        ax.plot(alphas, q_delta, 'o-', color=colors['Q'], linewidth=2, 
                markersize=8, label='Q-scaling')
        ax.plot(alphas, k_delta, 's--', color=colors['K'], linewidth=2,
                markersize=8, label='K-scaling')
        
        ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        ax.set_xlabel('α', fontsize=11)
        ax.set_ylabel('ΔH', fontsize=11)
        ax.set_title(titles[idx], fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xticks(alphas)
    
    plt.suptitle('EXP3b: Gain-Conditioned Q vs K Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# INTERPRETATION FUNCTIONS
# ============================================================

def compute_asymmetry_metric(df: pd.DataFrame) -> Dict:
    """
    Compute Q-K asymmetry metric.
    
    Returns dict with:
    - q_range: range of Q entropy
    - k_range: range of K entropy
    - asymmetry_ratio: q_range / k_range
    """
    q_data = df[df['scaling_type'] == 'Q']['entropy']
    k_data = df[df['scaling_type'] == 'K']['entropy']
    
    q_range = q_data.max() - q_data.min()
    k_range = k_data.max() - k_data.min()
    
    asymmetry = q_range / k_range if k_range > 0 else float('inf')
    
    return {
        'q_range': q_range,
        'k_range': k_range,
        'asymmetry_ratio': asymmetry,
        'is_q_dominant': asymmetry > 1.2,
        'is_symmetric': 0.8 <= asymmetry <= 1.2
    }


def interpret_result(head_name: str, metrics: Dict) -> str:
    """
    Generate interpretation following strict rules.
    
    No free interpretation allowed.
    """
    if 'high_gain' in head_name:
        if metrics['is_q_dominant']:
            return "Query-dominant control (Q ≠ K)"
        else:
            return "Generic logit scaling (Q ≈ K)"
    
    elif 'medium_gain' in head_name:
        if metrics['is_q_dominant']:
            return "Partial Q-dominance (transitional regime)"
        elif metrics['is_symmetric']:
            return "Generic regime (Q ≈ K)"
        else:
            return "K-biased (unexpected)"
    
    else:  # low_gain
        if metrics['is_symmetric']:
            return "Generic baseline confirmed (Q ≈ K) — negative control passed"
        else:
            return "Unexpected asymmetry in low-gain head"


# ============================================================
# MAIN EXECUTION
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="EXP3b: Gain-Conditioned Q vs K Scaling"
    )
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("EXP3b — GAIN-CONDITIONED Q vs K SCALING")
    print("=" * 70)
    print("\nFINAL EXPERIMENT IN SPECTRA")
    print("\nScientific question:")
    print("  Does Q–K asymmetry emerge selectively in high-gain heads?")
    print("\nHeads tested (LOCKED):")
    for name, info in TARGET_HEADS.items():
        print(f"  {name}: Layer {info['layer']}, Head {info['head']} (gain ≈ {info['expected_gain']})")
    print(f"\nAlphas: {ALPHAS}")
    print("=" * 70)
    
    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    
    print(f"\n--- GPU ---")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"\n--- Loading Model ---")
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    print(f"  Model: {args.model}")
    print(f"  Load time: {time.time() - start:.1f}s")
    
    # Load sample
    print(f"\n--- Loading Sample ---")
    samples = load_from_shards(args.data_dir, n_samples=args.sample_idx + 1, device="cuda")
    if samples and len(samples) > args.sample_idx:
        input_ids = samples[args.sample_idx]["input_ids"]
    else:
        sample = load_long_context(tokenizer, target_length=512)
        input_ids = sample["input_ids"]
    print(f"  Tokens: {input_ids.shape[1]}")
    
    # Create profiler
    profiler = Exp3bProfiler(model)
    
    # Capture hidden states once
    profiler.register_hooks()
    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
    
    # Run comparisons
    print("\n" + "=" * 70)
    print("RUNNING COMPARISONS")
    print("=" * 70)
    
    all_results = {}
    all_metrics = {}
    
    try:
        for name, info in TARGET_HEADS.items():
            df = profiler.run_head_comparison(
                input_ids,
                info['layer'],
                info['head'],
                ALPHAS,
                name,
                verbose=True
            )
            all_results[name] = df
            all_metrics[name] = compute_asymmetry_metric(df)
            
            # Save individual results
            head_dir = OUTPUT_DIR / f"L{info['layer']}_H{info['head']}"
            head_dir.mkdir(exist_ok=True)
            
            df.to_csv(head_dir / "qk_results.csv", index=False)
            plot_entropy_comparison(df, head_dir / "qk_entropy.png", name)
            plot_delta_entropy(df, head_dir / "qk_delta_entropy.png", name)
            
            torch.cuda.empty_cache()
    
    finally:
        profiler.remove_hooks()
    
    # Combined plot
    print("\n--- Generating Combined Plot ---")
    plot_combined_comparison(all_results, OUTPUT_DIR / "exp3b_combined.png")
    
    # Save combined results
    combined_df = pd.concat(all_results.values(), ignore_index=True)
    combined_df.to_csv(OUTPUT_DIR / "exp3b_all_results.csv", index=False)
    
    # Print interpretation
    print("\n" + "=" * 70)
    print("RESULTS & INTERPRETATION")
    print("=" * 70)
    
    for name, metrics in all_metrics.items():
        info = TARGET_HEADS[name]
        interpretation = interpret_result(name, metrics)
        
        print(f"\n--- {name} (L{info['layer']} H{info['head']}) ---")
        print(f"  Q entropy range: {metrics['q_range']:.4f}")
        print(f"  K entropy range: {metrics['k_range']:.4f}")
        print(f"  Asymmetry ratio: {metrics['asymmetry_ratio']:.2f}")
        print(f"  → {interpretation}")
    
    # Summary table
    summary_data = []
    for name, metrics in all_metrics.items():
        info = TARGET_HEADS[name]
        summary_data.append({
            'head_name': name,
            'layer': info['layer'],
            'head': info['head'],
            'expected_gain': info['expected_gain'],
            'q_range': metrics['q_range'],
            'k_range': metrics['k_range'],
            'asymmetry_ratio': metrics['asymmetry_ratio'],
            'interpretation': interpret_result(name, metrics)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(OUTPUT_DIR / "exp3b_summary.csv", index=False)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nKey files:")
    print(f"  Combined plot: exp3b_combined.png")
    print(f"  Summary: exp3b_summary.csv")
    print(f"  Per-head: L*_H*/qk_delta_entropy.png (PRIMARY FIGURES)")
    
    print("\n" + "=" * 70)
    print("SPECTRA EXPERIMENTAL PHASE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
