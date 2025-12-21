# Spectra: An Adaptive Inference Method for Transformer Attention

## Phase 1: Query Norm → Attention Entropy Correlation Study

**Goal**: Empirically verify whether Query Norm (‖Q‖) predicts Attention Entropy across layers and heads in a pretrained LLM.

### Hypothesis

- **H₀**: corr(‖Q‖, entropy) = 0
- **H₁**: corr(‖Q‖, entropy) < 0 (negative correlation expected)

### Experimental Configuration

| Parameter      | Value               |
| -------------- | ------------------- |
| Model          | Llama-3-8B          |
| Precision      | fp16                |
| Context Length | 4K tokens (initial) |
| Batch Size     | 1                   |
| Mode           | Inference only      |

### Metrics

1. **Query Norm**: `‖Q_{ℓ,h,t}‖₂` — L2 norm of query vector per layer, head, token
2. **Attention Entropy**: `H = -Σ pᵢ log(pᵢ)` — entropy of attention distribution

### Success Criteria

- |r| ≥ 0.5 correlation in meaningful subset of heads
- p-value < 0.01
- Randomization control shows ~0 correlation

## Project Structure

```
Spectra/
├── src/
│   ├── __init__.py
│   ├── config.py           # Experiment configuration
│   ├── model_loader.py     # Model loading utilities
│   ├── hooks.py            # Attention hooks for Q and probs
│   ├── metrics.py          # Query norm & entropy computation
│   ├── data_loader.py      # Long-context data preparation
│   └── analysis.py         # Statistical analysis
├── scripts/
│   ├── run_experiment.py   # Main experiment script
│   └── visualize.py        # Visualization generation
├── notebooks/              # Jupyter notebooks for exploration
├── data/                   # Input data (gitignored)
├── results/                # Experiment outputs (gitignored)
├── requirements.txt        # Python dependencies
└── README.md
```

## Environment Setup

### Prerequisites

- NVIDIA GPU (≥ 24 GB VRAM)
- CUDA 12.1+
- Conda

### Installation

```bash
# Create environment
conda create -n phase1_llm python=3.10 -y
conda activate phase1_llm

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### Verify CUDA

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

## Running on SageMaker

This project is designed to run on AWS SageMaker with GPU instances. All code is written for CUDA devices only.

## License

MIT
