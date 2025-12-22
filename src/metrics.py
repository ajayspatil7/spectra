"""
Metrics Computation for Phase 1 Experiment
============================================

Computes query norms, attention entropy, and correlations
for the Query Norm → Entropy correlation study.

Usage:
    from src.metrics import compute_query_norm, compute_attention_entropy, compute_correlations
"""

import torch
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class CorrelationResult:
    """Result of correlation analysis for a single (layer, head) pair."""
    layer: int
    head: int
    pearson_r: float
    pearson_p: float
    spearman_r: float
    spearman_p: float
    n_samples: int
    
    def is_significant(self, r_threshold: float = 0.5, p_threshold: float = 0.01) -> bool:
        """Check if correlation meets significance criteria."""
        return (
            abs(self.pearson_r) >= r_threshold and 
            self.pearson_p < p_threshold and
            not np.isnan(self.pearson_r)
        )


def compute_query_norm(Q: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 norm of query vectors.
    
    Args:
        Q: Query tensor [batch, n_heads, seq_len, head_dim]
           or [n_heads, seq_len, head_dim]
           
    Returns:
        norms: [batch, n_heads, seq_len] or [n_heads, seq_len]
    """
    return torch.norm(Q.float(), p=2, dim=-1)


def compute_attention_entropy(
    attn_probs: torch.Tensor,
    causal_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-9,
    ignore_first_n: int = 2
) -> torch.Tensor:
    """
    Compute entropy of attention distribution (mask-aware, NaN-safe).
    
    H = -Σ p_i * log(p_i)
    
    Higher entropy = more uniform attention (diffuse)
    Lower entropy = more focused attention (sparse)
    
    Args:
        attn_probs: [batch, n_heads, seq_len, seq_len] attention probabilities
        causal_mask: Optional [seq_len, seq_len] mask (True = masked)
        eps: Small constant for numerical stability
        ignore_first_n: Set entropy to NaN for first N tokens (insufficient context)
        
    Returns:
        entropy: [batch, n_heads, seq_len]
    """
    # Promote to float32 for numerical stability
    attn_probs = attn_probs.float()
    
    # Get dimensions
    if attn_probs.dim() == 4:
        batch, n_heads, seq_len, _ = attn_probs.shape
    else:
        n_heads, seq_len, _ = attn_probs.shape
        batch = None
    
    # Build causal mask if not provided
    if causal_mask is None:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=attn_probs.device),
            diagonal=1
        ).bool()
    
    # Valid mask: True where attention is allowed
    valid_mask = ~causal_mask
    if batch is not None:
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
    else:
        valid_mask = valid_mask.unsqueeze(0)  # [1, seq, seq]
    
    # Zero out masked positions
    masked_probs = attn_probs * valid_mask
    
    # Compute entropy only over valid positions
    # H = -Σ p * log(p), treating 0 * log(0) = 0
    log_probs = torch.log(masked_probs + eps)
    entropy = -torch.sum(masked_probs * log_probs, dim=-1)
    
    # Mark early tokens as NaN (insufficient context for meaningful entropy)
    if ignore_first_n > 0:
        if batch is not None:
            entropy[..., :ignore_first_n] = float('nan')
        else:
            entropy[:, :ignore_first_n] = float('nan')
    
    return entropy


def compute_max_attention_weight(
    attn_probs: torch.Tensor,
    causal_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute maximum attention weight per token.
    
    For each query token, returns the maximum attention weight it assigns
    to any key token in the valid attention range.
    
    Args:
        attn_probs: [batch, n_heads, seq_len, seq_len] attention probabilities
        causal_mask: Optional [seq_len, seq_len] mask (True = masked)
        
    Returns:
        max_weights: [batch, n_heads, seq_len]
    """
    attn_probs = attn_probs.float()
    
    # Get dimensions
    if attn_probs.dim() == 4:
        batch, n_heads, seq_len, _ = attn_probs.shape
    else:
        n_heads, seq_len, _ = attn_probs.shape
        batch = None
    
    # Build causal mask if not provided
    if causal_mask is None:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=attn_probs.device),
            diagonal=1
        ).bool()
    
    # Valid mask: True where attention is allowed - broadcast to match attn_probs shape
    valid_mask = ~causal_mask  # [seq, seq]
    
    # Expand to match attention probs dimensions
    # attn_probs is [batch, n_heads, seq, seq] or [n_heads, seq, seq]
    if batch is not None:
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
    else:
        valid_mask = valid_mask.unsqueeze(0)  # [1, seq, seq]
    
    # Broadcast will handle the n_heads dimension automatically
    # Mask out invalid positions
    masked_probs = attn_probs.masked_fill(~valid_mask, 0.0)
    
    # Get max along key dimension
    max_weights = torch.max(masked_probs, dim=-1).values
    
    return max_weights


def compute_effective_attention_span(
    attn_probs: torch.Tensor,
    causal_mask: Optional[torch.Tensor] = None,
    threshold: float = 0.9
) -> torch.Tensor:
    """
    Compute effective attention span (k_eff).
    
    Returns the minimum number of top-weighted keys needed to account
    for 'threshold' fraction (default 90%) of total attention mass.
    
    Args:
        attn_probs: [batch, n_heads, seq_len, seq_len] attention probabilities
        causal_mask: Optional [seq_len, seq_len] mask (True = masked)
        threshold: Cumulative probability threshold (default 0.9 for 90%)
        
    Returns:
        k_eff: [batch, n_heads, seq_len] effective span
    """
    attn_probs = attn_probs.float()
    
    # Get dimensions
    if attn_probs.dim() == 4:
        batch, n_heads, seq_len, _ = attn_probs.shape
    else:
        n_heads, seq_len, _ = attn_probs.shape
        batch = None
    
    # Build causal mask if not provided
    if causal_mask is None:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=attn_probs.device),
            diagonal=1
        ).bool()
    
    # Valid mask - broadcast to match attn_probs shape
    valid_mask = ~causal_mask  # [seq, seq]
    
    # Expand to match attention probs dimensions
    if batch is not None:
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
    else:
        valid_mask = valid_mask.unsqueeze(0)  # [1, seq, seq]
    
    # Broadcast will handle the n_heads dimension
    # Zero out masked positions
    masked_probs = attn_probs * valid_mask.float()
    
    # Sort attention weights in descending order
    sorted_probs, _ = torch.sort(masked_probs, dim=-1, descending=True)
    
    # Compute cumulative sum
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    
    # Find first position where cumsum >= threshold
    # Add 1 because we want count (1-indexed), not index (0-indexed)
    k_eff = torch.sum(cumsum < threshold, dim=-1) + 1
    
    return k_eff.float()



def compute_correlation_single(
    q_norms: np.ndarray,
    entropy: np.ndarray
) -> Tuple[float, float, float, float, int]:
    """
    Compute Pearson and Spearman correlations between query norm and entropy.
    
    Args:
        q_norms: 1D array of query norms
        entropy: 1D array of entropy values
        
    Returns:
        Tuple of (pearson_r, pearson_p, spearman_r, spearman_p, n_samples)
    """
    # Remove NaN values
    valid = ~(np.isnan(q_norms) | np.isnan(entropy))
    q_norms = q_norms[valid]
    entropy = entropy[valid]
    
    n_samples = len(q_norms)
    
    # Need at least 4 samples for meaningful correlation
    if n_samples < 4:
        return np.nan, np.nan, np.nan, np.nan, n_samples
    
    # Compute correlations
    try:
        pearson_r, pearson_p = stats.pearsonr(q_norms, entropy)
        spearman_r, spearman_p = stats.spearmanr(q_norms, entropy)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, n_samples
    
    return pearson_r, pearson_p, spearman_r, spearman_p, n_samples


def compute_correlations(
    q_norms: torch.Tensor,
    entropy: torch.Tensor,
    verbose: bool = True
) -> List[CorrelationResult]:
    """
    Compute correlations for all (layer, head) pairs.
    
    Args:
        q_norms: [n_layers, n_heads, seq_len] query norms
        entropy: [n_layers, n_heads, seq_len] entropy values
        verbose: Whether to print progress
        
    Returns:
        List of CorrelationResult for each (layer, head) pair
    """
    n_layers, n_heads, seq_len = q_norms.shape
    
    # Convert to numpy
    q_norms_np = q_norms.cpu().float().numpy()
    entropy_np = entropy.cpu().float().numpy()
    
    results = []
    significant_count = 0
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            # Get values for this (layer, head)
            norms = q_norms_np[layer_idx, head_idx]
            ent = entropy_np[layer_idx, head_idx]
            
            # Compute correlation
            pearson_r, pearson_p, spearman_r, spearman_p, n = compute_correlation_single(norms, ent)
            
            result = CorrelationResult(
                layer=layer_idx,
                head=head_idx,
                pearson_r=pearson_r,
                pearson_p=pearson_p,
                spearman_r=spearman_r,
                spearman_p=spearman_p,
                n_samples=n
            )
            results.append(result)
            
            if result.is_significant():
                significant_count += 1
    
    if verbose:
        total = n_layers * n_heads
        print(f"Computed correlations for {total} (layer, head) pairs")
        print(f"Significant correlations (|r| >= 0.5, p < 0.01): {significant_count}/{total} ({100*significant_count/total:.1f}%)")
    
    return results


def results_to_dataframe(results: List[CorrelationResult]) -> pd.DataFrame:
    """Convert correlation results to a pandas DataFrame."""
    data = []
    for r in results:
        data.append({
            'layer': r.layer,
            'head': r.head,
            'pearson_r': r.pearson_r,
            'pearson_p': r.pearson_p,
            'spearman_r': r.spearman_r,
            'spearman_p': r.spearman_p,
            'n_samples': r.n_samples,
            'significant': r.is_significant()
        })
    return pd.DataFrame(data)


def results_to_heatmap_matrix(
    results: List[CorrelationResult],
    n_layers: int,
    n_heads: int,
    metric: str = 'pearson_r'
) -> np.ndarray:
    """
    Convert results to a 2D matrix for heatmap visualization.
    
    Args:
        results: List of correlation results
        n_layers: Number of layers
        n_heads: Number of heads
        metric: Which metric to use ('pearson_r', 'spearman_r', etc.)
        
    Returns:
        2D numpy array of shape [n_layers, n_heads]
    """
    matrix = np.full((n_layers, n_heads), np.nan)
    
    for r in results:
        matrix[r.layer, r.head] = getattr(r, metric)
    
    return matrix


def run_randomization_control(
    q_norms: torch.Tensor,
    entropy: torch.Tensor,
    n_permutations: int = 100,
    seed: int = 42
) -> Dict:
    """
    Randomization control: shuffle entropy and recompute correlations.
    
    If the relationship is real, shuffled correlations should collapse to ~0.
    
    Args:
        q_norms: [n_layers, n_heads, seq_len] query norms
        entropy: [n_layers, n_heads, seq_len] entropy values
        n_permutations: Number of random permutations
        seed: Random seed
        
    Returns:
        Dict with null distribution statistics
    """
    np.random.seed(seed)
    
    n_layers, n_heads, seq_len = q_norms.shape
    
    # Convert to numpy
    q_norms_np = q_norms.cpu().float().numpy()
    entropy_np = entropy.cpu().float().numpy()
    
    # Collect shuffled correlations
    shuffled_rs = []
    
    print(f"Running {n_permutations} randomization permutations...")
    
    for perm in range(n_permutations):
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                norms = q_norms_np[layer_idx, head_idx]
                ent = entropy_np[layer_idx, head_idx].copy()
                
                # Shuffle entropy values
                valid = ~np.isnan(ent)
                ent[valid] = np.random.permutation(ent[valid])
                
                # Compute correlation
                r, _, _, _, _ = compute_correlation_single(norms, ent)
                if not np.isnan(r):
                    shuffled_rs.append(r)
    
    shuffled_rs = np.array(shuffled_rs)
    
    result = {
        'mean_shuffled_r': np.mean(shuffled_rs),
        'std_shuffled_r': np.std(shuffled_rs),
        'abs_mean_shuffled_r': np.mean(np.abs(shuffled_rs)),
        'max_abs_shuffled_r': np.max(np.abs(shuffled_rs)),
        'n_permutations': n_permutations,
        'n_samples': len(shuffled_rs),
    }
    
    print(f"Shuffled correlations: mean={result['mean_shuffled_r']:.4f}, "
          f"std={result['std_shuffled_r']:.4f}, "
          f"|mean|={result['abs_mean_shuffled_r']:.4f}")
    
    return result


def print_summary(results: List[CorrelationResult]):
    """Print a summary of correlation results."""
    df = results_to_dataframe(results)
    
    print("=" * 70)
    print("CORRELATION SUMMARY")
    print("=" * 70)
    
    # Overall statistics
    valid_pearson = df['pearson_r'].dropna()
    print(f"\nPearson r statistics:")
    print(f"  Mean:   {valid_pearson.mean():+.4f}")
    print(f"  Std:    {valid_pearson.std():.4f}")
    print(f"  Min:    {valid_pearson.min():+.4f}")
    print(f"  Max:    {valid_pearson.max():+.4f}")
    
    # Significance counts
    sig = df['significant'].sum()
    total = len(df)
    print(f"\nSignificant correlations (|r| >= 0.5, p < 0.01):")
    print(f"  {sig}/{total} ({100*sig/total:.1f}%)")
    
    # By layer
    print(f"\nMean |r| by layer:")
    layer_means = df.groupby('layer')['pearson_r'].apply(lambda x: np.abs(x).mean())
    for layer, mean_r in layer_means.items():
        if layer % 8 == 0 or layer == len(layer_means) - 1:  # Every 8th layer
            print(f"  Layer {layer:2d}: {mean_r:.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Test the metrics functions
    print("Testing metrics module...")
    
    # Create dummy data
    n_layers, n_heads, seq_len = 4, 8, 100
    
    q_norms = torch.randn(n_layers, n_heads, seq_len).abs()
    entropy = torch.randn(n_layers, n_heads, seq_len).abs()
    entropy[:, :, :2] = float('nan')  # Simulate masked early tokens
    
    # Test correlation computation
    results = compute_correlations(q_norms, entropy)
    print_summary(results)
    
    # Test heatmap matrix
    matrix = results_to_heatmap_matrix(results, n_layers, n_heads)
    print(f"Heatmap matrix shape: {matrix.shape}")
    
    # Test randomization control
    null_dist = run_randomization_control(q_norms, entropy, n_permutations=10)
    
    print("✅ Metrics test passed!")
