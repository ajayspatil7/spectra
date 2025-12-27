"""
Entropy Computation for MATS 10.0
==================================

Computes Shannon entropy of attention distributions for sycophancy analysis.

Key Metrics:
- Per-token entropy: H(t) = -Σ p(i|t) * log(p(i|t))
- ΔEntropy: Difference between Sycophancy and Control conditions
- Entropy statistics: Mean, std, per-head aggregations

Interpretation:
- High entropy = Diffuse attention = Head is "uncertain"
- Low entropy = Focused attention = Head is "confident"

Hypothesis Mapping:
- Logic Heads: High ΔEntropy during rationalization (blur to bridge truth-lie gap)
- Sycophancy Heads: Low ΔEntropy during rationalization (sharpen on hint)
"""

from typing import Tuple, Dict, List, Optional
import torch
import numpy as np


def calculate_entropy(
    pattern: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute per-token Shannon entropy over attention distribution.
    
    LOCKED DEFINITION (for reproducibility):
    H(t) = -Σ_{i=0}^{t} p(i|t) * log(p(i|t))
    
    Args:
        pattern: Attention pattern [batch, head, query_pos, key_pos]
                 or [head, query_pos, key_pos] for single sample
        eps: Small constant for numerical stability
        
    Returns:
        Entropy tensor [batch, head, query_pos] or [head, query_pos]
        
    Example:
        logits, cache = model.run_with_cache(prompt)
        pattern = cache["pattern", layer]  # [batch, head, q, k]
        entropy = calculate_entropy(pattern)  # [batch, head, q]
    """
    # Clamp to avoid log(0)
    pattern = torch.clamp(pattern, min=eps, max=1.0)
    
    # Shannon entropy: H = -Σ p * log(p)
    log_pattern = torch.log(pattern)
    entropy = -torch.sum(pattern * log_pattern, dim=-1)
    
    return entropy


def calculate_entropy_stats(
    pattern: torch.Tensor,
    ignore_first_n: int = 2,
    eps: float = 1e-10,
) -> Dict[str, torch.Tensor]:
    """
    Compute entropy with statistics (mean, std) per head.
    
    Args:
        pattern: Attention pattern [batch, head, query_pos, key_pos]
        ignore_first_n: Ignore first N positions (BOS artifacts)
        eps: Numerical stability constant
        
    Returns:
        Dict with:
            - "entropy": Full entropy tensor
            - "mean": Mean entropy per head
            - "std": Std entropy per head
    """
    entropy = calculate_entropy(pattern, eps)
    
    # Mask first N positions (BOS/padding)
    if ignore_first_n > 0:
        entropy[..., :ignore_first_n] = float('nan')
    
    # Aggregate across positions (nanmean to ignore NaN)
    mean = torch.nanmean(entropy, dim=-1)
    
    # Manual nanstd (torch doesn't have built-in)
    valid_mask = ~torch.isnan(entropy)
    entropy_masked = torch.where(valid_mask, entropy, torch.zeros_like(entropy))
    count = valid_mask.sum(dim=-1).float()
    std = torch.sqrt(
        (torch.sum(valid_mask * (entropy_masked - mean.unsqueeze(-1))**2, dim=-1) / count)
    )
    
    return {
        "entropy": entropy,
        "mean": mean,
        "std": std,
    }


def compute_delta_entropy(
    pattern_sycophancy: torch.Tensor,
    pattern_control: torch.Tensor,
    ignore_first_n: int = 2,
) -> Dict[str, torch.Tensor]:
    """
    Compute ΔEntropy between Sycophancy and Control conditions.
    
    HANDLES VARIABLE-LENGTH SEQUENCES: Compares mean entropy per head,
    not per-position delta (since prompts have different lengths).
    
    ΔE = mean(E_sycophancy) - mean(E_control)
    
    Interpretation:
    - ΔE > 0: Head becomes MORE diffuse during rationalization
    - ΔE < 0: Head becomes MORE focused during rationalization
    
    Args:
        pattern_sycophancy: Attention under sycophancy prompt [batch, head, q, k]
        pattern_control: Attention under control prompt [batch, head, q, k]
        ignore_first_n: Ignore first N positions
        
    Returns:
        Dict with:
            - "mean_entropy_syco": Mean entropy per head [batch, head]
            - "mean_entropy_ctrl": Mean entropy per head [batch, head]
            - "mean_delta": Mean delta per head [batch, head]
    """
    # Compute entropy for each condition separately
    entropy_syco = calculate_entropy(pattern_sycophancy)  # [batch, head, seq_syco]
    entropy_ctrl = calculate_entropy(pattern_control)     # [batch, head, seq_ctrl]
    
    # Mask early positions
    if ignore_first_n > 0:
        entropy_syco[..., :ignore_first_n] = float('nan')
        entropy_ctrl[..., :ignore_first_n] = float('nan')
    
    # Compute mean entropy per head (handles different seq lengths)
    mean_syco = torch.nanmean(entropy_syco, dim=-1)  # [batch, head]
    mean_ctrl = torch.nanmean(entropy_ctrl, dim=-1)  # [batch, head]
    
    # Delta of means (not mean of deltas - sequences are different lengths)
    mean_delta = mean_syco - mean_ctrl
    
    return {
        "mean_entropy_syco": mean_syco,
        "mean_entropy_ctrl": mean_ctrl,
        "mean_delta": mean_delta,
    }


def compute_head_entropy_profile(
    cache,
    target_layers: List[int],
    n_heads: int = 28,
    ignore_first_n: int = 2,
) -> Dict[Tuple[int, int], float]:
    """
    Compute mean entropy for each (layer, head) pair from cache.
    
    Args:
        cache: TransformerLens activation cache
        target_layers: List of layer indices to analyze
        n_heads: Number of attention heads per layer
        ignore_first_n: Ignore first N positions
        
    Returns:
        Dict mapping (layer, head) -> mean entropy
    """
    profile = {}
    
    for layer in target_layers:
        pattern = cache["pattern", layer]  # [batch, head, q, k]
        entropy = calculate_entropy(pattern)  # [batch, head, q]
        
        # Mask and aggregate
        if ignore_first_n > 0:
            entropy[..., :ignore_first_n] = float('nan')
        
        mean_per_head = torch.nanmean(entropy, dim=(0, -1))  # [head]
        
        for head in range(n_heads):
            profile[(layer, head)] = mean_per_head[head].item()
    
    return profile


def identify_head_types(
    delta_entropy_results: List[Dict[Tuple[int, int], float]],
    logic_threshold: float = 0.5,
    sycophancy_threshold: float = -0.3,
    consistency_pct: float = 0.7,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Identify Logic Heads and Sycophancy Heads based on ΔEntropy patterns.
    
    Logic Heads: ΔE > +logic_threshold in ≥consistency_pct of problems
    Sycophancy Heads: ΔE < sycophancy_threshold in ≥consistency_pct of problems
    
    Args:
        delta_entropy_results: List of dicts, each mapping (layer, head) -> ΔE
        logic_threshold: Threshold for Logic Head classification
        sycophancy_threshold: Threshold for Sycophancy Head classification
        consistency_pct: Min fraction of problems where pattern must hold
        
    Returns:
        Tuple of (logic_heads, sycophancy_heads) as lists of (layer, head)
    """
    if not delta_entropy_results:
        return [], []
    
    # Collect all (layer, head) pairs
    all_heads = set()
    for result in delta_entropy_results:
        all_heads.update(result.keys())
    
    logic_heads = []
    sycophancy_heads = []
    n_problems = len(delta_entropy_results)
    
    for lh in all_heads:
        # Count how often this head exceeds thresholds
        logic_count = sum(1 for r in delta_entropy_results if r.get(lh, 0) > logic_threshold)
        syco_count = sum(1 for r in delta_entropy_results if r.get(lh, 0) < sycophancy_threshold)
        
        if logic_count / n_problems >= consistency_pct:
            logic_heads.append(lh)
        if syco_count / n_problems >= consistency_pct:
            sycophancy_heads.append(lh)
    
    # Sort by layer, then head
    logic_heads.sort()
    sycophancy_heads.sort()
    
    return logic_heads, sycophancy_heads


def entropy_to_numpy(entropy_dict: Dict) -> Dict[str, np.ndarray]:
    """Convert entropy dict to numpy for plotting/saving."""
    return {
        k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
        for k, v in entropy_dict.items()
    }
