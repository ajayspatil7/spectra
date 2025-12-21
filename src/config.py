"""
Experiment configuration for Phase 1.

All hyperparameters are frozen before running experiments.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Frozen configuration for Phase 1 experiment."""
    
    # Model configuration
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    precision: str = "fp16"  # Options: fp16, bf16
    device: str = "cuda"  # CUDA only
    
    # Context configuration
    max_context_length: int = 4096  # Start with 4K
    batch_size: int = 1  # Single sample for cleaner analysis
    
    # Experiment configuration
    seed: int = 42
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    
    # Statistical thresholds (fixed before analysis)
    correlation_threshold: float = 0.5  # |r| >= 0.5 for significance
    p_value_threshold: float = 0.01  # p < 0.01 for significance
    
    def __post_init__(self):
        """Validate configuration."""
        # Enforce CUDA-only
        if self.device != "cuda":
            raise ValueError("This experiment requires CUDA. Set device='cuda'.")
        
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This experiment requires a GPU.\n"
                "Please run on a CUDA-enabled machine (e.g., SageMaker GPU instance)."
            )
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_torch_dtype(self) -> torch.dtype:
        """Get torch dtype from precision string."""
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        return dtype_map.get(self.precision, torch.float16)
    
    def get_device(self) -> torch.device:
        """Get torch device."""
        return torch.device(self.device)
    
    def print_gpu_info(self):
        """Print GPU information for verification."""
        print("=" * 50)
        print("GPU Configuration")
        print("=" * 50)
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("=" * 50)


# Default configuration
DEFAULT_CONFIG = ExperimentConfig()
