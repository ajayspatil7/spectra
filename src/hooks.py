"""
Attention Hooks for Phase 1 Experiment
=======================================

Instruments the model to capture Q vectors and attention probabilities
from all layers during a forward pass.

Usage:
    from src.hooks import AttentionProfiler
    
    profiler = AttentionProfiler(model)
    data = profiler.profile(input_ids)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class LayerData:
    """Data captured from a single attention layer."""
    query: torch.Tensor          # [batch, n_heads, seq_len, head_dim]
    key: torch.Tensor            # [batch, n_kv_heads, seq_len, head_dim]
    value: torch.Tensor          # [batch, n_kv_heads, seq_len, head_dim]
    attn_probs: Optional[torch.Tensor] = None  # [batch, n_heads, seq_len, seq_len]


class AttentionProfiler:
    """
    Captures Q, K, V vectors and attention probabilities from all layers.
    
    This class uses forward hooks to intercept internal tensors during
    the forward pass without modifying model outputs.
    
    IMPORTANT: Pre-RoPE Analysis
    ============================
    This profiler captures Q and K vectors BEFORE Rotary Position Embedding
    (RoPE) is applied. The attention probabilities computed here use:
    
        attention = softmax(Q @ K^T / sqrt(d))
    
    NOT the model's actual:
    
        attention = softmax(RoPE(Q) @ RoPE(K)^T / sqrt(d))
    
    Implications:
    - Query norms (‖Q‖) are measured in the pre-rotational space
    - Attention entropy reflects pre-rotational attention patterns
    - This is a deliberate choice for Phase 1: we analyze the INTRINSIC
      geometry of query vectors before positional encoding
    
    This is valid because:
    1. RoPE preserves vector norms (it's a rotation)
    2. We're testing whether Q magnitude predicts attention diffusion
    3. The pre-RoPE space is position-agnostic, which may be desirable
    
    For Phase 2, consider comparing pre-RoPE vs post-RoPE correlations.
    """
    
    def __init__(self, model):
        """
        Initialize the profiler.
        
        Args:
            model: HuggingFace causal LM model (e.g., LlamaForCausalLM)
        """
        self.model = model
        self.config = model.config
        
        # Get architecture parameters from config
        self.n_layers = self.config.num_hidden_layers
        self.n_heads = self.config.num_attention_heads
        self.n_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.hidden_size // self.n_heads
        self.hidden_size = self.config.hidden_size
        
        # Storage for captured data
        self.layer_data: Dict[int, LayerData] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Intermediate storage during forward pass
        self._hidden_states: Dict[int, torch.Tensor] = {}
        
    def _get_qkv_hook(self, layer_idx: int):
        """
        Create a hook to capture Q, K, V projections.
        
        We hook into the layer's input_layernorm output to get the
        normalized hidden states, then manually compute Q, K, V.
        """
        def hook_fn(module, input, output):
            # output is the normalized hidden states before attention
            # We'll use this to compute Q, K, V manually
            self._hidden_states[layer_idx] = output.detach()
        return hook_fn
    
    def _compute_qkv(self, layer_idx: int, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Manually compute Q, K, V from hidden states.
        
        This gives us access to the raw projections before RoPE.
        """
        layer = self.model.model.layers[layer_idx]
        attn = layer.self_attn
        
        batch_size, seq_len, _ = hidden_states.shape
        
        with torch.no_grad():
            # Project to Q, K, V
            Q = attn.q_proj(hidden_states)
            K = attn.k_proj(hidden_states)
            V = attn.v_proj(hidden_states)
            
            # Reshape Q: [batch, seq, n_heads * head_dim] -> [batch, n_heads, seq, head_dim]
            Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            
            # Reshape K, V: [batch, seq, n_kv_heads * head_dim] -> [batch, n_kv_heads, seq, head_dim]
            K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        return Q, K, V
    
    def _compute_attention_probs(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor,
        causal: bool = True
    ) -> torch.Tensor:
        """
        Compute attention probabilities from Q and K.
        
        IMPORTANT: Recomputation vs Capture
        ====================================
        We RECOMPUTE attention here rather than capturing internal model attention.
        
        Why not capture directly?
        - HuggingFace's LlamaAttention uses FlashAttention/SDPA by default
        - These optimized kernels don't expose intermediate attention weights
        - Enabling output_attentions=True disables these optimizations
        
        Implications of recomputation:
        - No dropout (model uses dropout=0.0 for inference anyway)
        - Standard softmax (matches model behavior)
        - Pre-RoPE attention (see class docstring)
        - May have minor numerical differences from optimized kernels
        
        For Phase 1 correlation study, this is acceptable because:
        1. We're measuring relative relationships (correlation), not exact values
        2. The Q norms are from the same pre-RoPE space as our attention
        3. Consistency between Q and attention computation is maintained
        
        Args:
            Q: [batch, n_heads, seq, head_dim]
            K: [batch, n_kv_heads, seq, head_dim]
            causal: Whether to apply causal masking
            
        Returns:
            attn_probs: [batch, n_heads, seq, seq]
        """
        batch_size, n_heads, seq_len, head_dim = Q.shape
        _, n_kv_heads, _, _ = K.shape
        
        # Expand K for GQA if needed
        n_rep = n_heads // n_kv_heads
        if n_rep > 1:
            K = K.repeat_interleave(n_rep, dim=1)
        
        # Compute attention scores: [batch, n_heads, seq, seq]
        scale = head_dim ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        
        # Apply causal mask
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax to get probabilities
        attn_probs = torch.softmax(scores, dim=-1)
        
        return attn_probs
    
    def register_hooks(self):
        """Register forward hooks on all attention layers."""
        self.remove_hooks()  # Clean up any existing hooks
        
        for layer_idx in range(self.n_layers):
            layer = self.model.model.layers[layer_idx]
            
            # Hook into input_layernorm to get pre-attention hidden states
            hook = layer.input_layernorm.register_forward_hook(
                self._get_qkv_hook(layer_idx)
            )
            self.hooks.append(hook)
        
        print(f"Registered hooks on {len(self.hooks)} layers")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._hidden_states = {}
    
    def clear_data(self):
        """Clear all captured data."""
        self.layer_data = {}
        self._hidden_states = {}
    
    def profile(
        self, 
        input_ids: torch.Tensor,
        compute_attn_probs: bool = True
    ) -> Dict[int, LayerData]:
        """
        Run a forward pass and capture attention data from all layers.
        
        Args:
            input_ids: [batch, seq_len] input token IDs
            compute_attn_probs: Whether to compute attention probabilities
            
        Returns:
            Dict mapping layer_idx to LayerData
        """
        self.clear_data()
        self.register_hooks()
        
        try:
            # Run forward pass to trigger hooks
            with torch.no_grad():
                _ = self.model(input_ids, use_cache=False)
            
            # Process captured hidden states
            print(f"Processing {len(self._hidden_states)} layers...")
            
            for layer_idx, hidden_states in self._hidden_states.items():
                # Compute Q, K, V
                Q, K, V = self._compute_qkv(layer_idx, hidden_states)
                
                # Optionally compute attention probs
                attn_probs = None
                if compute_attn_probs:
                    attn_probs = self._compute_attention_probs(Q, K, causal=True)
                
                # Store layer data
                self.layer_data[layer_idx] = LayerData(
                    query=Q,
                    key=K,
                    value=V,
                    attn_probs=attn_probs
                )
            
            print(f"✅ Captured data from {len(self.layer_data)} layers")
            
        finally:
            self.remove_hooks()
        
        return self.layer_data
    
    def get_all_query_norms(self) -> torch.Tensor:
        """
        Get query norms from all layers.
        
        Returns:
            Tensor of shape [n_layers, n_heads, seq_len]
        """
        if not self.layer_data:
            raise RuntimeError("No data captured. Run profile() first.")
        
        norms = []
        for layer_idx in sorted(self.layer_data.keys()):
            Q = self.layer_data[layer_idx].query
            # L2 norm along head_dim: [batch, n_heads, seq] -> [n_heads, seq]
            layer_norms = torch.norm(Q, p=2, dim=-1).squeeze(0)
            norms.append(layer_norms)
        
        return torch.stack(norms)  # [n_layers, n_heads, seq]
    
    def get_all_attention_entropy(self, causal_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Get attention entropy from all layers using the canonical implementation.
        
        Uses compute_attention_entropy from metrics.py to ensure consistency.
        
        Args:
            causal_mask: Optional [seq_len, seq_len] causal mask (True = masked).
                         If None, will be constructed automatically.
        
        Returns:
            Tensor of shape [n_layers, n_heads, seq_len]
        """
        # Import here to avoid circular imports
        from src.metrics import compute_attention_entropy
        
        if not self.layer_data:
            raise RuntimeError("No data captured. Run profile() first.")
        
        entropies = []
        for layer_idx in sorted(self.layer_data.keys()):
            attn_probs = self.layer_data[layer_idx].attn_probs
            if attn_probs is None:
                raise RuntimeError(f"No attention probs for layer {layer_idx}")
            
            # Use the single canonical entropy implementation
            # Note: ignore_first_n defaults to 2 in compute_attention_entropy
            entropy = compute_attention_entropy(
                attn_probs,
                causal_mask=causal_mask
            ).squeeze(0)  # [n_heads, seq]
            
            entropies.append(entropy)
        
        return torch.stack(entropies)  # [n_layers, n_heads, seq]
    
    def print_summary(self):
        """Print a summary of captured data."""
        if not self.layer_data:
            print("No data captured yet.")
            return
        
        print("=" * 60)
        print("PROFILER SUMMARY")
        print("=" * 60)
        print(f"Layers captured: {len(self.layer_data)}")
        
        sample = self.layer_data[0]
        print(f"Q shape: {sample.query.shape}")
        print(f"K shape: {sample.key.shape}")
        print(f"V shape: {sample.value.shape}")
        if sample.attn_probs is not None:
            print(f"Attn probs shape: {sample.attn_probs.shape}")
        
        # Memory usage
        total_bytes = 0
        for data in self.layer_data.values():
            total_bytes += data.query.element_size() * data.query.nelement()
            total_bytes += data.key.element_size() * data.key.nelement()
            total_bytes += data.value.element_size() * data.value.nelement()
            if data.attn_probs is not None:
                total_bytes += data.attn_probs.element_size() * data.attn_probs.nelement()
        
        print(f"Total captured data: {total_bytes / 1e9:.2f} GB")
        print("=" * 60)


if __name__ == "__main__":
    # Test the profiler
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print("Testing AttentionProfiler...")
    
    # Load model
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # Create profiler
    profiler = AttentionProfiler(model)
    
    # Test with short input
    inputs = tokenizer("Hello, world!", return_tensors="pt").to("cuda")
    
    # Profile
    data = profiler.profile(inputs.input_ids)
    
    # Print summary
    profiler.print_summary()
    
    # Get aggregated metrics
    q_norms = profiler.get_all_query_norms()
    entropy = profiler.get_all_attention_entropy()
    
    print(f"\nQuery norms shape: {q_norms.shape}")
    print(f"Entropy shape: {entropy.shape}")
    print("✅ Profiler test passed!")
