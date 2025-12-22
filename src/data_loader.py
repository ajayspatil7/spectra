"""
Data Loader for Phase 1 Experiment
===================================

Prepares long-context input data for the Query Norm → Entropy correlation study.

Usage:
    from src.data_loader import load_long_context
    
    inputs = load_long_context(tokenizer, target_length=4096)
"""

import torch
from typing import Dict, Optional
from pathlib import Path


# Sample long text for experiments (diverse content)
SAMPLE_TEXT = """
The history of artificial intelligence began in antiquity, with myths, stories and rumors of 
artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of 
modern AI were planted by classical philosophers who attempted to describe the process of human 
thinking as the mechanical manipulation of symbols. This work culminated in the invention of the 
programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical 
reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously 
discussing the possibility of building an electronic brain.

The field of AI research was founded at a workshop held on the campus of Dartmouth College during 
the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of 
them predicted that a machine as intelligent as a human being would exist in no more than a 
generation, and they were given millions of dollars to make this vision come true. Eventually, it 
became obvious that they had grossly underestimated the difficulty of the project.

In 1973, in response to the criticism of James Lighthill and ongoing pressure from the US Congress 
to fund more productive projects, both the U.S. and British Governments cut off exploratory research 
in AI. The next few years would later be called an "AI winter", a period when obtaining funding for 
AI projects was difficult. In the early 1980s, AI research was revived by the commercial success of 
expert systems, a form of AI program that simulated the knowledge and analytical skills of human 
experts. By 1985, the market for AI had reached over a billion dollars.

At the same time, Japan's fifth generation computer project inspired the U.S and British governments 
to restore funding for academic research. However, beginning with the collapse of the Lisp Machine 
market in 1987, AI once again fell into disrepute, and a second, longer-lasting winter began. Many 
researchers began to doubt that the symbolic approach would be able to imitate all the processes of 
human cognition, especially perception, robotics, learning and pattern recognition.

A number of researchers began to look into "sub-symbolic" approaches to specific AI problems. 
Robotics researchers, such as Rodney Brooks, rejected symbolic AI and focused on the basic 
engineering problems that would allow robots to move and survive. Their work revived the 
non-symbolic viewpoint of the early cybernetics researchers of the 1950s and reintroduced the use 
of control theory in AI. This coincided with the development of the embodied mind thesis in the 
related field of cognitive science: the idea that aspects of the body (such as movement, perception 
and visualization) are required for higher intelligence.

Similarly, some computer vision and robotics researchers argued that artificial intelligence 
required the development of artificial life. These approaches share the same basic idea: that 
intelligence is an emergent property that arises from the interaction between an agent and its 
environment. The problem of acquiring knowledge and handling uncertainty has been addressed using 
probability theory and economics.

The 1990s and early 21st century brought statistical approaches to AI, which exploited the 
increasing availability of data and processing power. Deep learning began to dominate industry 
benchmarks and in 2012, a deep learning architecture won the ImageNet Large Scale Visual Recognition 
Challenge for the first time. This success would lead to an explosion in research and funding of 
deep learning, and eventually to the development of large language models like GPT.

Transformers, introduced in 2017, revolutionized natural language processing. The attention mechanism
allowed models to process sequences in parallel and capture long-range dependencies more effectively
than recurrent neural networks. This architecture became the foundation for models like BERT, GPT,
and their successors. The scaling of transformer models led to emergent capabilities: abilities that
appear suddenly at certain scales of compute, data, and parameters.

The relationship between attention patterns and model behavior remains an active area of research.
Understanding how attention mechanisms distribute focus across input sequences can provide insights
into model interpretability and efficiency. Query, key, and value projections form the core of the
attention mechanism, with the query vector determining what information to attend to, the key vector
representing the information available, and the value vector containing the actual content.

In modern large language models, Grouped Query Attention (GQA) has emerged as an efficient variant
that reduces memory usage by sharing key-value heads across multiple query heads. Models like
Llama use RoPE (Rotary Position Embeddings) to encode positional information through rotation of
query and key vectors, enabling better extrapolation to longer sequences than absolute position
embeddings allowed.

The entropy of attention distributions provides a measure of how focused or diffuse the model's
attention is at each position. Low entropy indicates concentrated attention on few positions, while
high entropy suggests more uniform distribution across the context. Understanding the relationship
between query norms and attention entropy could lead to more efficient inference strategies that
adapt computational resources based on the complexity of attention patterns.

Memory efficiency in transformer inference has become increasingly important as models scale to
billions of parameters. Techniques like flash attention optimize memory access patterns, while
methods like speculative decoding aim to reduce latency. Understanding the predictive power of
query norms for attention complexity could enable new forms of adaptive computation.

The theoretical foundations of attention mechanisms draw from information retrieval concepts,
where queries retrieve relevant information from a database of keys and values. The softmax
operation ensures that attention weights form a probability distribution, and the temperature
parameter can control the sharpness of this distribution. In self-attention, each position
can attend to all previous positions (in causal/autoregressive models) or all positions
(in bidirectional models like BERT).

Multi-head attention allows the model to jointly attend to information from different
representation subspaces at different positions. Each head can learn to focus on different
aspects of the input: syntactic structure, semantic relationships, or positional patterns.
The outputs of all heads are concatenated and linearly projected to produce the final output.

Layer normalization, typically applied before attention (pre-norm) in modern architectures,
stabilizes training by normalizing activations. RMSNorm, used in Llama, simplifies this by
removing the mean-centering step. The feed-forward network following attention typically
expands the hidden dimension by a factor of 4 (or uses SwiGLU with different ratios) before
projecting back down, allowing for more complex computations.

Residual connections enable gradient flow through deep networks and allow each layer to learn
a refinement of its input rather than a complete transformation. This architectural choice,
combined with careful initialization and learning rate schedules, enables training of models
with hundreds of layers. The interplay between residual connections and layer normalization
affects how information flows through the network.

Understanding the internal representations of large language models remains challenging.
Probing tasks attempt to understand what information is encoded in hidden states, while
mechanistic interpretability aims to reverse-engineer the algorithms implemented by trained
networks. The relationship between low-level metrics like query norms and high-level behaviors
like attention entropy represents an intermediate level of analysis that could bridge these
approaches.
"""


def load_long_context(
    tokenizer,
    target_length: int = 4096,
    text: Optional[str] = None,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Prepare long-context input for experiments.
    
    Args:
        tokenizer: HuggingFace tokenizer
        target_length: Target sequence length in tokens
        text: Optional custom text (uses SAMPLE_TEXT if None)
        device: Device to place tensors on
        
    Returns:
        Dict with 'input_ids' and 'attention_mask' tensors
    """
    if text is None:
        text = SAMPLE_TEXT
    
    # Repeat text if needed to reach target length
    while True:
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        if tokens.input_ids.shape[1] >= target_length:
            break
        text = text + "\n\n" + text  # Double the text
    
    # Tokenize with truncation to exact length
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=target_length,
        padding=False
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    actual_length = inputs["input_ids"].shape[1]
    print(f"Prepared input: {actual_length} tokens on {device}")
    
    return inputs


def load_from_file(
    tokenizer,
    file_path: str,
    target_length: int = 4096,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Load text from a file and prepare as input.
    
    Args:
        tokenizer: HuggingFace tokenizer
        file_path: Path to text file
        target_length: Target sequence length
        device: Device to place tensors on
        
    Returns:
        Dict with 'input_ids' and 'attention_mask' tensors
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")
    
    text = path.read_text(encoding="utf-8")
    return load_long_context(tokenizer, target_length, text, device)


def load_from_dataset(
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "test",
    target_length: int = 4096,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Load text from a HuggingFace dataset.
    
    Args:
        tokenizer: HuggingFace tokenizer
        dataset_name: Dataset name on HuggingFace
        dataset_config: Dataset configuration
        split: Dataset split
        target_length: Target sequence length
        device: Device to place tensors on
        
    Returns:
        Dict with 'input_ids' and 'attention_mask' tensors
    """
    from datasets import load_dataset
    
    print(f"Loading dataset: {dataset_name}/{dataset_config} ({split})")
    dataset = load_dataset(dataset_name, dataset_config, split=split)
    
    # Concatenate text samples until we have enough
    text = ""
    for sample in dataset:
        text += sample.get("text", "") + "\n"
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        if tokens.input_ids.shape[1] >= target_length:
            break
    
    return load_long_context(tokenizer, target_length, text, device)


def load_from_shards(
    data_dir: str = "data/processed",
    n_samples: int = 1,
    device: str = "cuda"
) -> list:
    """
    Load samples from preprocessed .npz shards.
    
    Args:
        data_dir: Directory containing shard_*.npz files
        n_samples: Number of samples to load
        device: Device to place tensors on
        
    Returns:
        List of dicts, each containing 'input_ids' and 'attention_mask' tensors
    """
    import numpy as np
    
    data_path = Path(data_dir)
    
    # Find all shard files
    shard_files = sorted(data_path.glob("shard_*.npz"))
    
    if not shard_files:
        print(f"No shard files found in {data_dir}")
        print("Falling back to single sample mode")
        return None
    
    print(f"Found {len(shard_files)} shard files")
    
    # Load samples
    samples = []
    loaded = 0
    
    for shard_file in shard_files:
        if loaded >= n_samples:
            break
            
        # Load shard
        data = np.load(shard_file)
        input_ids = data["input_ids"]
        
        # Get samples from this shard
        for i in range(len(input_ids)):
            if loaded >= n_samples:
                break
            
            # Convert to tensor
            input_id_tensor = torch.from_numpy(input_ids[i]).unsqueeze(0).to(device)
            attention_mask_tensor = torch.ones_like(input_id_tensor)
            
            samples.append({
                "input_ids": input_id_tensor,
                "attention_mask": attention_mask_tensor
            })
            
            loaded += 1
    
    print(f"Loaded {loaded} samples from {data_dir}")
    
    return samples


if __name__ == "__main__":
    # Test the data loader
    from transformers import AutoTokenizer
    
    print("Testing data loader...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    # Test with sample text
    inputs = load_long_context(tokenizer, target_length=4096, device="cuda")
    
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    print(f"Attention mask shape: {inputs['attention_mask'].shape}")
    print(f"Device: {inputs['input_ids'].device}")
    print("✅ Data loader test passed!")
