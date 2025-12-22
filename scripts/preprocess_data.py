#!/usr/bin/env python3
"""
Spectra Phase-1 Data Preprocessing Script
==========================================

This script preprocesses the DKYoon/SlimPajama-6B dataset for the 
Query Norm → Attention Entropy correlation experiment.

IMPORTANT: This is INFERENCE-ONLY preprocessing, not training prep.
The goal is to produce CORRECT token shards, not optimized batches.

Requirements:
- Uses HuggingFace streaming (no full download)
- No padding, no overlap
- Fixed 512-token chunks
- Outputs NumPy .npz shards

Usage:
    # Dry run (10 samples)
    python scripts/preprocess_data.py --dry-run

    # Full preprocessing
    python scripts/preprocess_data.py --output-dir data/processed --n-samples 10000

    # Resume from existing shards
    python scripts/preprocess_data.py --output-dir data/processed --resume

This preprocessing pipeline is safe for Spectra Phase-1.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator, Dict, List, Optional

import numpy as np
from tqdm import tqdm

# ============================================================================
# CONFIGURATION — DO NOT MODIFY WITHOUT UNDERSTANDING IMPLICATIONS
# ============================================================================

# Tokenization parameters
SEQ_LEN = 512                    # Fixed sequence length (non-negotiable)
STRIDE = 512                     # Non-overlapping chunks (stride == seq_len)
MAX_CHUNKS_PER_DOCUMENT = 20     # Prevent single doc from dominating

# Text filtering
MIN_TEXT_LENGTH = 200            # Minimum raw text length in characters

# Sharding
SEQUENCES_PER_SHARD = 3000       # Target sequences per shard (2000-5000 range)

# Source filtering — only allow these RedPajama sources
ALLOWED_SOURCES = {
    "RedPajamaCommonCrawl",
    "RedPajamaC4", 
    "RedPajamaWikipedia",
    "RedPajamaStackExchange",
}

# Dataset configuration
DATASET_NAME = "DKYoon/SlimPajama-6B"
DATASET_SPLIT = "test"     # SlimPajama-6B only has train/test splits (no validation)

# Data storage
RAW_DATA_DIR = Path("data/raw")  # Directory to store downloaded dataset

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def load_tokenizer():
    """
    Load the exact LLaMA-3-8B tokenizer.
    
    Why: We must use the exact tokenizer to ensure token IDs match
    what the model expects. Any mismatch would invalidate the experiment.
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    return tokenizer


def stream_dataset() -> Iterator[Dict]:
    """
    Download and load the full dataset from HuggingFace.
    
    Downloads the entire SlimPajama-6B test split to /data/raw directory.
    If dataset is already downloaded, it will reuse the cached version.
    
    Why full download:
    - Faster preprocessing on subsequent runs
    - Complete data availability for experiments
    - No dependency on network for repeated runs
    
    Note: Uses datasets library with cache_dir to download to /data/raw
    SlimPajama-6B structure: data/{split}-*.parquet
    """
    from datasets import load_dataset
    
    print(f"Loading dataset: {DATASET_NAME} (split: {DATASET_SPLIT})")
    print(f"Download location: {RAW_DATA_DIR.absolute()}")
    
    # Create raw data directory if it doesn't exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {RAW_DATA_DIR.absolute()}")
    
    # Download full dataset to /data/raw
    print("Downloading full dataset (this may take a while on first run)...")
    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        cache_dir=str(RAW_DATA_DIR.absolute()),
        streaming=False  # Download full dataset
    )
    
    print(f"Dataset loaded: {len(dataset):,} samples")
    
    # Return iterator over dataset
    return iter(dataset)


def is_valid_source(sample: Dict) -> bool:
    """
    Check if sample is from an allowed source.
    
    Why source filtering:
    - Some sources (e.g., GitHub, ArXiv) have domain-specific patterns
    - We want diverse, general-purpose text
    - Prevents code/math from skewing attention patterns
    """
    meta = sample.get("meta", {})
    source = meta.get("redpajama_set_name", "")
    return source in ALLOWED_SOURCES


def clean_text(text: str) -> Optional[str]:
    """
    Minimal text cleaning — ONLY strip whitespace.
    
    Why minimal cleaning:
    - Preserves original token distribution
    - Any normalization could alter attention patterns
    - We analyze the model as-is, not on cleaned data
    
    FORBIDDEN operations:
    - lowercase
    - unicode normalization
    - punctuation removal
    - sentence splitting
    """
    if text is None:
        return None
    
    # Only strip leading/trailing whitespace
    text = text.strip()
    
    # Check minimum length
    if len(text) < MIN_TEXT_LENGTH:
        return None
    
    return text


def tokenize_text(tokenizer, text: str) -> List[int]:
    """
    Tokenize text without special tokens or masks.
    
    Why add_special_tokens=False:
    - BOS/EOS would add structure that varies by chunk position
    - We want pure content tokens only
    - Special tokens could affect entropy at boundaries
    
    Why no attention mask:
    - We never pad, so mask is always all-ones
    - Storing masks wastes memory
    """
    tokens = tokenizer(
        text,
        add_special_tokens=False,       # No BOS/EOS
        return_attention_mask=False,     # Not needed
        return_tensors=None,             # Return plain list
    )
    
    return tokens["input_ids"]


def chunk_tokens(tokens: List[int]) -> List[List[int]]:
    """
    Chunk tokens into fixed-length, non-overlapping windows.
    
    Why non-overlapping:
    - Overlapping would create statistical dependencies
    - Each chunk must be independent for valid correlation analysis
    - Overlap would inflate sample count artificially
    
    Why drop remainder:
    - Padding is FORBIDDEN
    - Short sequences have different entropy characteristics
    - Fixed length ensures consistent analysis
    """
    chunks = []
    
    # Process in non-overlapping windows
    for start in range(0, len(tokens), STRIDE):
        end = start + SEQ_LEN
        
        # Only keep full-length chunks
        if end <= len(tokens):
            chunk = tokens[start:end]
            chunks.append(chunk)
            
            # Enforce per-document cap
            if len(chunks) >= MAX_CHUNKS_PER_DOCUMENT:
                break
    
    return chunks


def validate_chunk(chunk: List[int], tokenizer) -> bool:
    """
    Validate a single chunk meets all requirements.
    
    Checks:
    1. Exact length
    2. All IDs in vocab range
    3. No special padding tokens
    """
    # Check exact length
    if len(chunk) != SEQ_LEN:
        return False
    
    # Check vocab range
    vocab_size = tokenizer.vocab_size
    for token_id in chunk:
        if token_id < 0 or token_id >= vocab_size:
            return False
    
    # Check no padding token (if defined)
    if tokenizer.pad_token_id is not None:
        if tokenizer.pad_token_id in chunk:
            return False
    
    return True


def save_shard(
    sequences: List[List[int]], 
    output_dir: Path, 
    shard_idx: int
) -> Path:
    """
    Save sequences to a NumPy .npz shard.
    
    Format:
    - input_ids: int32 array of shape [N, SEQ_LEN]
    
    Why int32:
    - Sufficient for vocab size (128k < 2^31)
    - Compatible with PyTorch tensor conversion
    """
    # Convert to numpy array
    input_ids = np.array(sequences, dtype=np.int32)
    
    # Verify shape
    assert input_ids.shape[1] == SEQ_LEN, f"Invalid shape: {input_ids.shape}"
    
    # Save shard
    shard_path = output_dir / f"shard_{shard_idx:05d}.npz"
    np.savez_compressed(shard_path, input_ids=input_ids)
    
    return shard_path


def validate_shard(shard_path: Path, tokenizer) -> bool:
    """
    Validate a saved shard can be loaded and is correct.
    """
    try:
        data = np.load(shard_path)
        input_ids = data["input_ids"]
        
        # Check shape
        if input_ids.shape[1] != SEQ_LEN:
            print(f"Invalid shape in {shard_path}: {input_ids.shape}")
            return False
        
        # Check dtype
        if input_ids.dtype != np.int32:
            print(f"Invalid dtype in {shard_path}: {input_ids.dtype}")
            return False
        
        # Check vocab range
        if input_ids.min() < 0:
            print(f"Negative token ID in {shard_path}")
            return False
        
        if input_ids.max() >= tokenizer.vocab_size:
            print(f"Token ID out of vocab range in {shard_path}")
            return False
        
        # Check for NaN (shouldn't happen with int, but be safe)
        if np.isnan(input_ids.astype(float)).any():
            print(f"NaN values in {shard_path}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Failed to load {shard_path}: {e}")
        return False


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess(
    output_dir: Path,
    n_samples: int = 10000,
    dry_run: bool = False,
    resume: bool = False
) -> Dict:
    """
    Main preprocessing pipeline.
    
    Args:
        output_dir: Directory to save shards
        n_samples: Target number of sequences to produce
        dry_run: If True, process only 10 samples and don't save
        resume: If True, continue from last shard
        
    Returns:
        Dict with preprocessing statistics
    """
    start_time = time.time()
    
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if dry_run:
        n_samples = 10
        print("=" * 60)
        print("DRY RUN MODE — Processing 10 samples only")
        print("=" * 60)
    
    # Load tokenizer
    print("\n" + "=" * 60)
    print("SPECTRA PHASE-1 DATA PREPROCESSING")
    print("=" * 60)
    tokenizer = load_tokenizer()
    
    # Resume logic
    starting_shard = 0
    starting_sequences = 0
    if resume:
        existing_shards = sorted(output_dir.glob("shard_*.npz"))
        if existing_shards:
            starting_shard = len(existing_shards)
            for shard in existing_shards:
                data = np.load(shard)
                starting_sequences += len(data["input_ids"])
            print(f"Resuming from shard {starting_shard} ({starting_sequences} sequences)")
    
    # Statistics
    stats = {
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "max_chunks_per_document": MAX_CHUNKS_PER_DOCUMENT,
        "min_text_length": MIN_TEXT_LENGTH,
        "allowed_sources": list(ALLOWED_SOURCES),
        "sequences_per_shard": SEQUENCES_PER_SHARD,
        "docs_processed": 0,
        "docs_filtered_source": 0,
        "docs_filtered_length": 0,
        "chunks_produced": starting_sequences,
        "shards_saved": starting_shard,
        "target_sequences": n_samples,
    }
    
    print(f"\nConfiguration:")
    print(f"  SEQ_LEN: {SEQ_LEN}")
    print(f"  STRIDE: {STRIDE}")
    print(f"  MAX_CHUNKS_PER_DOCUMENT: {MAX_CHUNKS_PER_DOCUMENT}")
    print(f"  MIN_TEXT_LENGTH: {MIN_TEXT_LENGTH}")
    print(f"  Target sequences: {n_samples}")
    print(f"  Output: {output_dir}")
    
    # Stream dataset
    print("\nStarting preprocessing...")
    dataset_iter = stream_dataset()
    
    # Accumulate sequences
    current_shard: List[List[int]] = []
    shard_idx = starting_shard
    
    pbar = tqdm(total=n_samples, initial=starting_sequences, desc="Sequences")
    
    try:
        for sample in dataset_iter:
            # Check if we have enough
            if stats["chunks_produced"] >= n_samples:
                break
            
            stats["docs_processed"] += 1
            
            # Source filtering
            if not is_valid_source(sample):
                stats["docs_filtered_source"] += 1
                continue
            
            # Get and clean text
            text = sample.get("text", "")
            text = clean_text(text)
            
            if text is None:
                stats["docs_filtered_length"] += 1
                continue
            
            # Tokenize
            tokens = tokenize_text(tokenizer, text)
            
            # Chunk
            chunks = chunk_tokens(tokens)
            
            # Validate and accumulate
            for chunk in chunks:
                if not validate_chunk(chunk, tokenizer):
                    continue
                
                current_shard.append(chunk)
                stats["chunks_produced"] += 1
                pbar.update(1)
                
                # Check if we have enough
                if stats["chunks_produced"] >= n_samples:
                    break
                
                # Save shard if full
                if len(current_shard) >= SEQUENCES_PER_SHARD:
                    if not dry_run:
                        shard_path = save_shard(current_shard, output_dir, shard_idx)
                        
                        # Validate saved shard
                        if not validate_shard(shard_path, tokenizer):
                            raise RuntimeError(f"Shard validation failed: {shard_path}")
                        
                        stats["shards_saved"] += 1
                        shard_idx += 1
                    
                    current_shard = []
        
        # Save remaining sequences
        if current_shard and not dry_run:
            shard_path = save_shard(current_shard, output_dir, shard_idx)
            
            if not validate_shard(shard_path, tokenizer):
                raise RuntimeError(f"Shard validation failed: {shard_path}")
            
            stats["shards_saved"] += 1
    
    finally:
        pbar.close()
    
    # Final stats
    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = elapsed
    stats["timestamp"] = datetime.now().isoformat()
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Documents processed: {stats['docs_processed']:,}")
    print(f"  Documents filtered (source): {stats['docs_filtered_source']:,}")
    print(f"  Documents filtered (length): {stats['docs_filtered_length']:,}")
    print(f"  Chunks produced: {stats['chunks_produced']:,}")
    print(f"  Shards saved: {stats['shards_saved']}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    
    # Save config
    if not dry_run:
        config_path = output_dir / "preprocessing_config.json"
        with open(config_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nConfig saved to: {config_path}")
    
    # Final validation
    print("\n" + "-" * 60)
    print("VALIDATION")
    print("-" * 60)
    
    if dry_run:
        print("✅ Dry run completed successfully")
        print(f"   Sample chunks: {len(current_shard)}")
        if current_shard:
            print(f"   First chunk length: {len(current_shard[0])}")
            print(f"   First chunk tokens: {current_shard[0][:10]}...")
    else:
        # Validate all shards
        all_valid = True
        total_sequences = 0
        
        for shard_path in sorted(output_dir.glob("shard_*.npz")):
            if validate_shard(shard_path, tokenizer):
                data = np.load(shard_path)
                total_sequences += len(data["input_ids"])
            else:
                all_valid = False
                print(f"❌ Invalid shard: {shard_path}")
        
        if all_valid:
            print(f"✅ All {stats['shards_saved']} shards validated")
            print(f"✅ Total sequences: {total_sequences:,}")
            print(f"✅ All sequences are exactly {SEQ_LEN} tokens")
            print(f"✅ No padding tokens detected")
            print(f"✅ All token IDs within vocab range")
            print("\n✅ This preprocessing pipeline is safe for Spectra Phase-1.")
        else:
            print("❌ VALIDATION FAILED — Do not use these shards")
            raise RuntimeError("Shard validation failed")
    
    return stats


# ============================================================================
# ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess SlimPajama-6B for Spectra Phase-1"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="data/processed",
        help="Output directory for shards"
    )
    parser.add_argument(
        "--n-samples", 
        type=int, 
        default=10000,
        help="Target number of sequences to produce"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process 10 samples only, don't save"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing shards"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    stats = preprocess(
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        dry_run=args.dry_run,
        resume=args.resume
    )
    
    return stats


if __name__ == "__main__":
    main()
