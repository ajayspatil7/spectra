"""
Model Dissection: Exploring Llama-3-8B Architecture
====================================================

Interactive exploration of the model's internal structure.
Convert to .ipynb for cell-by-cell execution.

Sections:
1. Model Loading
2. Architecture Overview
3. Layer Inspection
4. Attention Mechanism Deep Dive
5. Weight Analysis
6. Forward Hook Demonstration
7. Token Flow Visualization
"""

# %% [markdown]
# # Model Dissection: Llama-3-8B Architecture
# 
# This notebook explores every component of the Llama model.

# %% [1] Imports and Setup
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import gc

# Enforce CUDA
assert torch.cuda.is_available(), "CUDA required!"
device = torch.device("cuda")
print(f"Using: {torch.cuda.get_device_name(0)}")

# %% [2] Load Model
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print("Model loaded!")

# %% [markdown]
# ---
# ## Section 1: Architecture Overview
# ---

# %% [3] Print Full Architecture
def print_model_architecture(model):
    """Print the complete model architecture."""
    print("=" * 80)
    print("COMPLETE MODEL ARCHITECTURE")
    print("=" * 80)
    print(model)
    print("=" * 80)

# Uncomment to see full architecture (very long output)
# print_model_architecture(model)

# %% [4] Model Configuration
def print_model_config(model):
    """Print key model configuration parameters."""
    config = model.config
    
    print("=" * 60)
    print("MODEL CONFIGURATION")
    print("=" * 60)
    
    key_params = [
        ("Model Type", config.model_type),
        ("Hidden Size (d_model)", config.hidden_size),
        ("Intermediate Size (FFN)", config.intermediate_size),
        ("Num Attention Heads", config.num_attention_heads),
        ("Num Key-Value Heads", config.num_key_value_heads),
        ("Num Hidden Layers", config.num_hidden_layers),
        ("Vocab Size", config.vocab_size),
        ("Max Position Embeddings", config.max_position_embeddings),
        ("RoPE Theta", getattr(config, 'rope_theta', 'N/A')),
        ("RMS Norm Epsilon", config.rms_norm_eps),
        ("Tie Word Embeddings", config.tie_word_embeddings),
    ]
    
    for name, value in key_params:
        print(f"{name:30} : {value}")
    
    print("=" * 60)
    return config

config = print_model_config(model)

# %% [5] Count Parameters
def count_parameters(model) -> Dict[str, int]:
    """Count parameters by component."""
    param_counts = {}
    
    for name, param in model.named_parameters():
        # Get top-level component
        parts = name.split('.')
        if len(parts) >= 2:
            component = f"{parts[0]}.{parts[1]}"
        else:
            component = parts[0]
        
        if component not in param_counts:
            param_counts[component] = 0
        param_counts[component] += param.numel()
    
    return param_counts

def print_parameter_summary(model):
    """Print parameter count summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("PARAMETER SUMMARY")
    print("=" * 60)
    print(f"Total Parameters:     {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Memory (fp16):        {total_params * 2 / 1e9:.2f} GB")
    print(f"Memory (fp32):        {total_params * 4 / 1e9:.2f} GB")
    print("=" * 60)
    
    # By component
    param_counts = count_parameters(model)
    print("\nBy Component:")
    print("-" * 60)
    for component, count in sorted(param_counts.items(), key=lambda x: -x[1]):
        percentage = count / total_params * 100
        print(f"{component:40} : {count:>15,} ({percentage:5.2f}%)")

print_parameter_summary(model)

# %% [markdown]
# ---
# ## Section 2: Layer Structure
# ---

# %% [6] Explore Model Layers
def explore_layer_structure(model):
    """Explore the structure of transformer layers."""
    print("=" * 60)
    print("TRANSFORMER LAYER STRUCTURE")
    print("=" * 60)
    
    # Get first layer
    layer = model.model.layers[0]
    
    print(f"\nLayer Type: {type(layer).__name__}")
    print(f"\nComponents in each layer:")
    print("-" * 40)
    
    for name, module in layer.named_children():
        print(f"\n  {name}: {type(module).__name__}")
        
        # Go one level deeper
        if hasattr(module, 'named_children'):
            for subname, submodule in module.named_children():
                print(f"    └── {subname}: {type(submodule).__name__}")
                
                # Get shapes if it's a linear layer
                if isinstance(submodule, nn.Linear):
                    print(f"        Shape: {submodule.weight.shape}")

explore_layer_structure(model)

# %% [7] Visualize Layer Dimensions
def visualize_layer_dimensions(model):
    """Create a visual representation of data flow through a layer."""
    config = model.config
    
    print("=" * 60)
    print("DATA FLOW THROUGH ONE TRANSFORMER LAYER")
    print("=" * 60)
    
    d_model = config.hidden_size
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    d_head = d_model // n_heads
    d_ff = config.intermediate_size
    
    flow = f"""
    Input: [batch, seq_len, {d_model}]
           │
           ▼
    ┌──────────────────────────────────────────────────┐
    │  INPUT LAYER NORM (RMSNorm)                      │
    │  Shape: [{d_model}] → [{d_model}]                │
    └──────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────┐
    │  SELF-ATTENTION                                  │
    │                                                  │
    │  Q projection: [{d_model}] → [{d_model}]         │
    │      Reshape: [{n_heads} heads × {d_head} dim]   │
    │                                                  │
    │  K projection: [{d_model}] → [{n_kv_heads * d_head}]  │
    │      (Grouped Query Attention: {n_kv_heads} KV heads) │
    │                                                  │
    │  V projection: [{d_model}] → [{n_kv_heads * d_head}]  │
    │                                                  │
    │  Attention: softmax(QK^T / √{d_head}) × V       │
    │                                                  │
    │  O projection: [{d_model}] → [{d_model}]         │
    └──────────────────────────────────────────────────┘
           │
           ▼
    + ────── (Residual Connection)
           │
           ▼
    ┌──────────────────────────────────────────────────┐
    │  POST-ATTENTION LAYER NORM (RMSNorm)             │
    │  Shape: [{d_model}] → [{d_model}]                │
    └──────────────────────────────────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────┐
    │  FEED-FORWARD NETWORK (SwiGLU)                   │
    │                                                  │
    │  gate_proj: [{d_model}] → [{d_ff}]               │
    │  up_proj:   [{d_model}] → [{d_ff}]               │
    │                                                  │
    │  hidden = SiLU(gate) × up                        │
    │                                                  │
    │  down_proj: [{d_ff}] → [{d_model}]               │
    └──────────────────────────────────────────────────┘
           │
           ▼
    + ────── (Residual Connection)
           │
           ▼
    Output: [batch, seq_len, {d_model}]
    """
    print(flow)
    
    print("\nKey Architecture Notes:")
    print("-" * 40)
    print(f"• Grouped Query Attention (GQA): {n_heads} Q heads, {n_kv_heads} KV heads")
    print(f"• KV sharing ratio: {n_heads // n_kv_heads}:1")
    print(f"• FFN expansion ratio: {d_ff / d_model:.1f}x")
    print(f"• Uses RMSNorm (not LayerNorm)")
    print(f"• Uses SwiGLU activation (not ReLU/GELU)")
    print(f"• Uses RoPE for positional encoding")

visualize_layer_dimensions(model)

# %% [markdown]
# ---
# ## Section 3: Attention Mechanism Deep Dive
# ---

# %% [8] Inspect Attention Module
def inspect_attention_module(model):
    """Deep dive into the attention mechanism."""
    print("=" * 60)
    print("ATTENTION MODULE INSPECTION")
    print("=" * 60)
    
    attn = model.model.layers[0].self_attn
    config = model.config
    
    print(f"\nAttention Module Type: {type(attn).__name__}")
    print(f"\nConfiguration (from model.config):")
    print("-" * 40)
    
    # Get values from config (more reliable than module attributes)
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // n_heads
    
    print(f"  num_attention_heads: {n_heads}")
    print(f"  num_key_value_heads: {n_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_key_value_groups: {n_heads // n_kv_heads}")
    
    print(f"\nProjection Weights:")
    print("-" * 40)
    print(f"  q_proj: {attn.q_proj.weight.shape} → Q vectors")
    print(f"  k_proj: {attn.k_proj.weight.shape} → K vectors")
    print(f"  v_proj: {attn.v_proj.weight.shape} → V vectors")
    print(f"  o_proj: {attn.o_proj.weight.shape} → Output projection")
    
    return attn

attn = inspect_attention_module(model)

# %% [9] Visualize Q, K, V Projection
def visualize_qkv_projections(model, tokenizer, text: str):
    """Visualize Q, K, V vectors for a sample input."""
    print("=" * 60)
    print("Q, K, V PROJECTION VISUALIZATION")
    print("=" * 60)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]
    
    print(f"\nInput: '{text}'")
    print(f"Tokens: {seq_len}")
    print(f"Token IDs: {input_ids[0].tolist()}")
    
    # Get embeddings
    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
    
    print(f"\nEmbedding shape: {hidden_states.shape}")
    
    # Get first layer attention and config
    attn = model.model.layers[0].self_attn
    config = model.config
    
    # Get head dimensions from config (not from attn module)
    n_heads = config.num_attention_heads
    n_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // n_heads
    
    # Apply layer norm first (important!)
    norm = model.model.layers[0].input_layernorm
    with torch.no_grad():
        normed = norm(hidden_states)
    
    # Project to Q, K, V
    with torch.no_grad():
        Q = attn.q_proj(normed)
        K = attn.k_proj(normed)
        V = attn.v_proj(normed)
    
    print(f"\nProjection Shapes (before reshape):")
    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")
    
    # Reshape to heads
    Q_reshaped = Q.view(1, seq_len, n_heads, head_dim).transpose(1, 2)
    K_reshaped = K.view(1, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    V_reshaped = V.view(1, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    
    print(f"\nReshaped to heads:")
    print(f"  Q: {Q_reshaped.shape} → [batch, {n_heads} heads, seq, {head_dim} dim]")
    print(f"  K: {K_reshaped.shape} → [batch, {n_kv_heads} KV heads, seq, {head_dim} dim]")
    print(f"  V: {V_reshaped.shape} → [batch, {n_kv_heads} KV heads, seq, {head_dim} dim]")
    
    return Q_reshaped, K_reshaped, V_reshaped

Q, K, V = visualize_qkv_projections(model, tokenizer, "Hello, world!")

# %% [10] Compute Attention Scores Manually
def compute_attention_manually(Q, K, V, layer_idx: int = 0):
    """Manually compute attention scores step by step."""
    print("=" * 60)
    print(f"MANUAL ATTENTION COMPUTATION (Layer {layer_idx})")
    print("=" * 60)
    
    # Get shapes
    batch, n_heads, seq_len, head_dim = Q.shape
    _, n_kv_heads, _, _ = K.shape
    
    # GQA: repeat K, V to match Q heads
    n_rep = n_heads // n_kv_heads
    if n_rep > 1:
        K = K.repeat_interleave(n_rep, dim=1)
        V = V.repeat_interleave(n_rep, dim=1)
        print(f"GQA: Repeated K,V {n_rep}x to match {n_heads} Q heads")
    
    # Step 1: QK^T
    print(f"\nStep 1: Compute QK^T")
    print(f"  Q shape: {Q.shape}")
    print(f"  K^T shape: {K.transpose(-2, -1).shape}")
    
    scores = torch.matmul(Q, K.transpose(-2, -1))
    print(f"  QK^T shape: {scores.shape}")
    
    # Step 2: Scale
    scale = head_dim ** 0.5
    print(f"\nStep 2: Scale by √d = √{head_dim} = {scale:.2f}")
    scaled_scores = scores / scale
    
    # Step 3: Causal mask
    print(f"\nStep 3: Apply causal mask")
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=Q.device), 
        diagonal=1
    ).bool()
    masked_scores = scaled_scores.masked_fill(causal_mask, float('-inf'))
    
    # Step 4: Softmax
    print(f"\nStep 4: Softmax → Attention probabilities")
    attn_probs = torch.softmax(masked_scores, dim=-1)
    print(f"  Attention probs shape: {attn_probs.shape}")
    print(f"  Sum per row (should be 1.0): {attn_probs[0, 0, :, :].sum(dim=-1)}")
    
    # Step 5: Weighted sum of V
    print(f"\nStep 5: Weighted sum of Values")
    output = torch.matmul(attn_probs, V)
    print(f"  Output shape: {output.shape}")
    
    return attn_probs, output, causal_mask


attn_probs, attn_output, causal_mask = compute_attention_manually(Q, K, V)


# %% [11] Visualize Attention Patterns
def visualize_attention_pattern(attn_probs, head_idx: int = 0, save_path: str = None):
    """Visualize attention pattern for a specific head."""
    # Get attention matrix for one head
    attn_matrix = attn_probs[0, head_idx].cpu().float().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_matrix, cmap='viridis', aspect='auto')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(f'Attention Pattern - Head {head_idx}')
    plt.colorbar(im, label='Attention Weight')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    return fig

# Uncomment to visualize:
# visualize_attention_pattern(attn_probs, head_idx=0)

# %% [markdown]
# ---
# ## Section 4: Query Norm & Entropy (Phase 1 Metrics)
# ---

# %% [12] Compute Query Norm
def compute_query_norms(Q):
    """
    Compute L2 norm of query vectors.
    
    This is the key metric for Phase 1: we hypothesize that
    ‖Q‖ predicts attention entropy.
    
    Args:
        Q: Query tensor [batch, n_heads, seq_len, head_dim]
        
    Returns:
        norms: [batch, n_heads, seq_len]
    """
    print("=" * 60)
    print("QUERY NORM COMPUTATION")
    print("=" * 60)
    
    # L2 norm along the head_dim axis
    norms = torch.norm(Q, p=2, dim=-1)
    
    print(f"Q shape: {Q.shape}")
    print(f"Norm shape: {norms.shape}")
    print(f"\nSample norms (head 0):")
    print(f"  Min: {norms[0, 0].min().item():.4f}")
    print(f"  Max: {norms[0, 0].max().item():.4f}")
    print(f"  Mean: {norms[0, 0].mean().item():.4f}")
    print(f"  Std: {norms[0, 0].std().item():.4f}")
    
    return norms

q_norms = compute_query_norms(Q)

# %% [13] Compute Attention Entropy (MASK-AWARE, NaN-SAFE)
def compute_attention_entropy(attn_probs, causal_mask=None, eps: float = 1e-9):
    """
    Mask-aware attention entropy.

    Correctly handles:
    - causal masking
    - zero-probability positions
    - fp16 instability

    Args:
        attn_probs: [batch, n_heads, seq_len, seq_len]
        causal_mask: [seq_len, seq_len] with True = masked
        eps: numerical stability constant

    Returns:
        entropy: [batch, n_heads, seq_len]
    """
    print("=" * 60)
    print("ATTENTION ENTROPY COMPUTATION (MASK-AWARE)")
    print("=" * 60)

    # Promote to fp32 for numerical stability
    attn_probs = attn_probs.float()

    batch, n_heads, seq_len, _ = attn_probs.shape

    # Build causal mask if not provided
    if causal_mask is None:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=attn_probs.device),
            diagonal=1
        ).bool()

    # valid_mask: True where attention is allowed
    valid_mask = ~causal_mask                     # [seq, seq]
    valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)  # [1,1,seq,seq]

    # Zero out masked probabilities explicitly
    masked_probs = attn_probs * valid_mask

    # Compute entropy ONLY over valid positions
    entropy = -torch.sum(
        masked_probs * torch.log(masked_probs + eps),
        dim=-1
    )

    # Optional: ignore trivial early tokens (no context)
    entropy[..., :2] = torch.nan

    print(f"Attention probs shape: {attn_probs.shape}")
    print(f"Entropy shape: {entropy.shape}")

    print("\nSample entropy (head 0):")
    finite_vals = entropy[0, 0][~torch.isnan(entropy[0, 0])]
    print(f"  Min: {finite_vals.min().item():.4f}")
    print(f"  Max: {finite_vals.max().item():.4f}")
    print(f"  Mean: {finite_vals.mean().item():.4f}")

    # Theoretical max entropy varies per token
    print("\nNote:")
    print("• Entropy is computed ONLY over valid causal support")
    print("• Early tokens are ignored (NaN by design)")
    print("• No log(0), no NaNs")

    return entropy


# %% [14] Correlate Query Norm with Entropy (NaN-safe)
def correlate_norm_entropy(q_norms, attn_entropy):
    """
    Compute correlation between query norm and attention entropy.
    NaN-safe and statistically correct.
    """
    from scipy import stats

    print("=" * 60)
    print("QUERY NORM ↔ ATTENTION ENTROPY CORRELATION (NaN-SAFE)")
    print("=" * 60)

    n_heads = q_norms.shape[1]

    print(f"\nPer-head correlations:")
    print("-" * 40)

    results = []

    for h in range(n_heads):
        norms = q_norms[0, h].cpu().float().numpy()
        entropy = attn_entropy[0, h].cpu().float().numpy()

        # ✅ DROP NaNs
        valid = ~np.isnan(entropy)
        norms = norms[valid]
        entropy = entropy[valid]

        # ⚠️ Need at least 4 points for correlation
        if len(norms) < 4:
            pearson_r = spearman_r = np.nan
            pearson_p = spearman_p = np.nan
        else:
            pearson_r, pearson_p = stats.pearsonr(norms, entropy)
            spearman_r, spearman_p = stats.spearmanr(norms, entropy)

        results.append({
            'head': h,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
        })

        if h < 5 or h >= n_heads - 2:
            print(
                f"  Head {h:2d}: "
                f"Pearson r={pearson_r:+.4f}, "
                f"Spearman ρ={spearman_r:+.4f}"
            )
        elif h == 5:
            print(f"  ... ({n_heads - 7} more heads)")

attn_entropy = compute_attention_entropy(attn_probs, causal_mask)    

# This will work with more tokens, but let's demonstrate structure
correlations = correlate_norm_entropy(q_norms, attn_entropy)

# %% [markdown]
# ---
# ## Section 5: Forward Hooks (How We'll Capture Data)
# ---

# %% [15] Hook Demonstration
class AttentionHook:
    """
    Hook to capture Q vectors and attention probabilities during forward pass.
    
    This is how we'll collect data for the full experiment.
    """
    
    def __init__(self):
        self.captured_data = {}
        self.hooks = []
    
    def create_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # Note: The exact structure depends on the model implementation
            # This is a template - adjust based on actual model outputs
            self.captured_data[layer_idx] = {
                'output': output,
                'module_type': type(module).__name__,
            }
            print(f"  Hook fired: Layer {layer_idx}")
        return hook_fn
    
    def register_hooks(self, model):
        """Register hooks on all attention layers."""
        print("Registering attention hooks...")
        
        for idx, layer in enumerate(model.model.layers):
            hook = layer.self_attn.register_forward_hook(self.create_hook(idx))
            self.hooks.append(hook)
        
        print(f"Registered {len(self.hooks)} hooks")
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        print("All hooks removed")
    
    def clear_data(self):
        """Clear captured data."""
        self.captured_data = {}

def demonstrate_hooks(model, tokenizer):
    """Demonstrate hook functionality."""
    print("=" * 60)
    print("FORWARD HOOK DEMONSTRATION")
    print("=" * 60)
    
    hook_manager = AttentionHook()
    hook_manager.register_hooks(model)
    
    # Run forward pass
    print("\nRunning forward pass...")
    inputs = tokenizer("Test", return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"\nCaptured data from {len(hook_manager.captured_data)} layers")
    
    # Cleanup
    hook_manager.remove_hooks()
    hook_manager.clear_data()
    
    return hook_manager

# Uncomment to test:
# hook_manager = demonstrate_hooks(model, tokenizer)

# %% [markdown]
# ---
# ## Section 6: Weight Statistics
# ---

# %% [16] Analyze Weight Distributions
def analyze_weight_distributions(model, layer_idx: int = 0):
    """Analyze weight distributions for a layer."""
    print("=" * 60)
    print(f"WEIGHT DISTRIBUTION ANALYSIS (Layer {layer_idx})")
    print("=" * 60)
    
    layer = model.model.layers[layer_idx]
    
    weight_stats = []
    for name, param in layer.named_parameters():
        if 'weight' in name:
            w = param.data.float().cpu().numpy().flatten()
            stats = {
                'name': name,
                'shape': tuple(param.shape),
                'mean': np.mean(w),
                'std': np.std(w),
                'min': np.min(w),
                'max': np.max(w),
            }
            weight_stats.append(stats)
    
    print(f"\n{'Name':<40} {'Shape':<20} {'Mean':>10} {'Std':>10}")
    print("-" * 80)
    for s in weight_stats:
        print(f"{s['name']:<40} {str(s['shape']):<20} {s['mean']:>10.4f} {s['std']:>10.4f}")
    
    return weight_stats

weight_stats = analyze_weight_distributions(model)

# %% [17] Visualize Weight Distribution
def plot_weight_histogram(model, layer_idx: int = 0, proj_name: str = "q_proj"):
    """Plot weight histogram for a specific projection."""
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    
    proj = getattr(attn, proj_name)
    weights = proj.weight.data.float().cpu().numpy().flatten()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(weights, bins=100, density=True, alpha=0.7)
    ax.set_xlabel('Weight Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Weight Distribution: Layer {layer_idx} {proj_name}')
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Uncomment to visualize:
# plot_weight_histogram(model, layer_idx=0, proj_name="q_proj")

# %% [markdown]
# ---
# ## Section 7: RoPE (Rotary Position Embedding)
# ---

# %% [18] Understand RoPE
def explain_rope(model):
    """Explain Rotary Position Embeddings."""
    print("=" * 60)
    print("ROTARY POSITION EMBEDDING (RoPE)")
    print("=" * 60)
    
    config = model.config
    
    print("""
    RoPE encodes position by rotating Q and K vectors.
    
    Key idea: 
    - Instead of adding position embeddings, we ROTATE the vectors
    - Rotation angle depends on position and dimension
    - Inner product Q·K naturally encodes relative position
    
    Benefits:
    - No absolute position limit (can extrapolate)
    - Relative position is captured
    - No learned position parameters needed
    """)
    
    print(f"\nRoPE Configuration:")
    print(f"  Base frequency (theta): {getattr(config, 'rope_theta', 10000)}")
    print(f"  Max position: {config.max_position_embeddings}")

explain_rope(model)

# %% [markdown]
# ---
# ## Section 8: Summary & Cleanup
# ---

# %% [19] Summary
def print_summary(model):
    """Print a summary of key findings."""
    config = model.config
    
    print("=" * 60)
    print("MODEL DISSECTION SUMMARY")
    print("=" * 60)
    
    print(f"""
    Model: {config._name_or_path}
    
    Architecture Highlights:
    ─────────────────────────
    • {config.num_hidden_layers} transformer layers
    • {config.hidden_size} hidden dimension
    • {config.num_attention_heads} attention heads
    • {config.num_key_value_heads} KV heads (GQA)
    • {config.intermediate_size} FFN dimension
    
    Key Components:
    ─────────────────────────
    • Attention: Grouped Query Attention (GQA)
    • Normalization: RMSNorm
    • Activation: SwiGLU
    • Position: RoPE
    
    Phase 1 Metrics:
    ─────────────────────────
    • Query Norm: ‖Q‖₂ computed per (layer, head, token)
    • Attention Entropy: H = -Σ p log(p)
    • Hypothesis: corr(‖Q‖, H) < 0
    
    Next Steps:
    ─────────────────────────
    1. Register hooks to capture Q and attention probs
    2. Run inference on long-context input
    3. Collect (layer, head, token, q_norm, entropy) data
    4. Compute Pearson/Spearman correlations
    5. Visualize results
    """)

print_summary(model)

# %% [20] Cleanup
def cleanup():
    """Clean up GPU memory."""
    global model, tokenizer, Q, K, V, attn_probs, attn_output, q_norms, attn_entropy
    
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU memory cleared.")

# Uncomment when done:
# cleanup()
