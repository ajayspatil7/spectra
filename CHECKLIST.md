# Spectra Phase 1 — Main Branch Checklist

**Branch:** `main` (baseline, protected)  
**Purpose:** Capture detailed per-token attention metrics from Llama-3-8B  
**Status:** ✅ COMPLETE — Moved to `intervene` branch for experiments

---

## ✅ Infrastructure Setup

### Repository & Environment

- [x] Git repository initialized and pushed to GitHub
- [x] `.gitignore` configured (excludes `data/`, `results/`, `.pkl`, etc.)
- [x] `requirements.txt` with PyTorch, transformers, scipy, pandas, tqdm
- [x] README.md with complete documentation
- [x] AWS SageMaker compatible (A10G GPU tested)

### Data Pipeline

- [x] **`scripts/preprocess_data.py`** — SlimPajama-6B preprocessing
  - Full dataset download to `/data/raw` (changed from streaming)
  - Source filtering: CommonCrawl, C4, Wikipedia, StackExchange
  - Excluded: GitHub, ArXiv, Books
  - Fixed 512-token non-overlapping chunks
  - NumPy `.npz` shard output (int32)
  - Validation checks for vocab range, length, no padding
- [x] **4 shards created** — 12,000 sequences total (~6.14M tokens)

---

## ✅ Core Implementation

### Attention Profiler (`src/hooks.py`)

- [x] **`AttentionProfiler` class** — Captures Q/K/V from all 32 layers
- [x] Forward hooks on `input_layernorm` per layer
- [x] **Pre-RoPE capture** — Queries before rotary position embedding
- [x] Manual attention recomputation: `softmax(Q @ K^T / sqrt(d))`
- [x] Causal masking applied
- [x] Hook registration and removal methods
- [x] Summary method for debugging

### Metrics Module (`src/metrics.py`)

- [x] **`compute_attention_entropy()`** — Shannon entropy of attention
  - Mask-aware, NaN-safe, float32 precision
  - First 2 tokens → NaN (insufficient context)
  - Formula: `H = -Σ p·log(p)`
- [x] **`compute_max_attention_weight()`** — Peak attention per token
  - Uses `masked_fill()` for proper broadcasting
- [x] **`compute_effective_attention_span()`** — k_eff metric
  - 90% cumulative attention mass threshold
  - Sorted descending, cumsum approach
- [x] **`compute_query_norm()`** — L2 norm of Q vectors

### Data Loader (`src/data_loader.py`)

- [x] **`load_long_context()`** — Single sample from hardcoded text
- [x] **`load_from_shards()`** — Multi-sample loading from .npz files
  - Sequential loading (first N samples)
  - Returns list of dicts with `input_ids`, `attention_mask`
  - Fallback to single sample if shards not found

---

## ✅ Experiment Script (`scripts/run_experiment.py`)

### Arguments

- [x] `--model` — Model name (default: meta-llama/Meta-Llama-3-8B)
- [x] `--context-length` — Sequence length (default: 512)
- [x] `--n-samples` — Number of samples to process (default: 1)
- [x] `--data-dir` — Shard directory (default: data/processed)
- [x] `--output-dir` — CSV output path (default: results)
- [x] `--seed` — Random seed

### Processing Loop

- [x] Load N samples from shards
- [x] For each sample:
  - Create `AttentionProfiler`
  - Run forward pass, capture Q/K/V/attention
  - Compute 4 metrics per (layer, head, token)
  - Append to CSV incrementally
  - Clear CUDA cache
- [x] Progress tracking with tqdm and ETA

### Output Format

- [x] CSV file: `attention_metrics_{timestamp}.csv`
- [x] Columns: `sample_id`, `layer`, `head`, `token_pos`, `query_norm`, `entropy`, `max_attn`, `k_eff`
- [x] 64 samples → 33.5M rows, ~768 MB

---

## ✅ Utility Scripts

### `scripts/explore_npz.py`

- [x] NPZ file investigation tool
- [x] Shows shape, dtype, statistics, unique values
- [x] Token frequency analysis

### `scripts/show_experiment_config.py`

- [x] Prints experiment configuration
- [x] Documents all design decisions
- [x] Displays metric formulas and thresholds

---

## ✅ Bug Fixes Applied

| Issue                                   | Fix                                                   | Commit    |
| --------------------------------------- | ----------------------------------------------------- | --------- |
| `CorrelationResult` missing `n_samples` | Added `n_samples=len(samples)`                        | `d3ca088` |
| Mask broadcasting error in metrics      | Changed to `masked_fill()` and `* valid_mask.float()` | `04d4681` |
| Streaming mode slow                     | Changed to full download with `cache_dir`             | `e03477c` |

---

## 📊 Experiment Runs Completed

### 64-Sample Capture (Main Result)

- **Samples:** 64 from SlimPajama-6B test split
- **Metrics:** query_norm, entropy, max_attn, k_eff
- **Rows:** 33,554,432
- **Runtime:** ~15-20 minutes on A10G
- **Output:** `results/attention_metrics_*.csv`

---

## 🔧 Configuration Summary

| Parameter        | Value                                     |
| ---------------- | ----------------------------------------- |
| Model            | meta-llama/Meta-Llama-3-8B                |
| Precision        | float16                                   |
| Layers           | 32                                        |
| Heads            | 32 (GQA: 8 KV heads)                      |
| Context          | 512 tokens                                |
| Attention        | Pre-RoPE, manually recomputed             |
| Entropy ignore   | First 2 tokens (NaN)                      |
| k_eff threshold  | 0.9 (90%)                                 |
| Sample selection | Sequential (not random/stratified)        |
| Data sources     | CommonCrawl, C4, Wikipedia, StackExchange |

---

## 📁 File Structure (Main Branch)

```
Spectra/
├── src/
│   ├── config.py               # Experiment configuration
│   ├── hooks.py                # AttentionProfiler (pre-RoPE)
│   ├── metrics.py              # All 4 metric functions
│   └── data_loader.py          # Shard loading
├── scripts/
│   ├── run_experiment.py       # Main capture script
│   ├── preprocess_data.py      # SlimPajama preprocessing
│   ├── explore_npz.py          # Shard explorer
│   └── show_experiment_config.py # Config display
├── data/
│   ├── raw/                    # Downloaded dataset (gitignored)
│   └── processed/              # .npz shards (gitignored)
├── results/                    # CSV outputs (gitignored)
├── CHECKLIST.md                # This file
├── README.md                   # Project documentation
└── .gitignore                  # Excludes data/results/pkl
```

---

## 🔀 Branch Information

| Branch      | Purpose                  | Status                 |
| ----------- | ------------------------ | ---------------------- |
| `main`      | Baseline metrics capture | ✅ Complete, protected |
| `intervene` | Phase 1 Experiment 1     | 🔄 Active development  |

---

## 📝 Key Design Decisions (Documented)

1. **Pre-RoPE Analysis**: Q/K captured before rotation — position-agnostic geometry
2. **Manual Attention**: Recomputed (not FlashAttention) — exposes intermediate weights
3. **Sequential Sampling**: First N samples from shards — deterministic, not stratified
4. **Incremental CSV**: Appends after each sample — prevents OOM on large runs
5. **No Correlation Analysis**: Removed — focus on raw data capture

---

## ✅ Ready for Next Phase

Main branch is complete and protected. All future experiments should:

1. Create new branch from `main`
2. Make experimental changes
3. Keep `main` as reproducible baseline

**Current active branch:** `intervene` (Phase 1 Experiment 1)

---

## 🧪 Phase-1 Experiment-1: Causal Intervention

### Objective

Test whether query magnitude functions as a **causal control signal** regulating attention sharpness.

### Implementation Status

- [x] **`src/intervention.py`** — InterventionProfiler class
  - Scales Q vector for single (layer, head) pair
  - Intervention after Q projection, before RoPE
  - Isolation verification test included
- [x] **`scripts/run_intervention.py`** — Main experiment script
  - Accepts target layer, head, and scale factors
  - Runs intervention sweep [0.5, 0.75, 1.0, 1.25, 1.5]
  - Computes entropy, max_attn, k_eff per scale
  - Saves results to CSV with trend analysis
- [x] **`scripts/plot_intervention.py`** — Visualization
  - Entropy vs Scale plot
  - Max Attention vs Scale plot
  - k_eff vs Scale plot
  - Combined 3-panel figure

### Expected Outcomes

| Metric        | Scale 0.5→1.5 | Expected Effect        |
| ------------- | ------------- | ---------------------- |
| Entropy       | —             | Decreasing (strongest) |
| Max Attention | —             | Increasing             |
| k_eff         | —             | Decreasing (weakest)   |

### Running the Experiment (SageMaker)

```bash
# Target head experiment
python scripts/run_intervention.py --target-layer 12 --target-head 0

# Generate plots
python scripts/plot_intervention.py --input results/intervention/q_scale_intervention_results.csv

# Control head (for comparison)
python scripts/run_intervention.py --target-layer 12 --target-head 15 --head-type control
```

### Results

| Run     | Layer | Head | Type   | Entropy Trend | Max Attn Trend | k_eff Trend |
| ------- | ----- | ---- | ------ | ------------- | -------------- | ----------- |
| Pending | 12    | 0    | target | —             | —              | —           |

---

### 🔴 Failure Criteria (Intellectual Honesty)

> **If the target head does not exhibit monotonic trends while the control head remains flat, the Phase-0 correlation is likely epiphenomenal.**

This means:

- The observed correlation between Q norm and entropy in Phase-0 was **coincidental**
- Q magnitude is **not** a causal control signal for this head
- The experiment honestly reports a null result

### 🔴 Validation Requirements (Non-Negotiable)

Before trusting results, verify:

1. **Baseline Sanity Check** — `no_intervention ≈ scale=1.0`

   - If these differ, the hook is introducing artifacts
   - Run: `run_baseline_sanity_check()` returns `True`

2. **Intervention Isolation** — Only target head Q is scaled

   - Other heads' Q unchanged
   - K, V vectors unchanged
   - Run: `verify_intervention_isolation()` returns `True`

3. **Directional Expectations Logged**
   - Script must print expected trends before running sweep
   - Forces explicit confrontation of unexpected behavior

---

## 📊 Path A: Gain Analysis

### Purpose

Quantify how strongly each attention head responds to query magnitude scaling.

> Does the target head exploit query magnitude with higher sensitivity than a random head?

### Implementation

- [x] **`notebooks/gain/compute_gain.py`** — Gain computation script
  - Loads intervention CSVs
  - Computes `Gain = slope of (max_attn vs log(scale))`
  - Fits linear regression via OLS
  - Outputs `gain_summary.csv` and comparison plots

### Metric Definition

```
Gain = d(mean_max_attn) / d(log(scale))
```

Higher Gain = higher sensitivity to Q magnitude.

### Running Gain Analysis

```bash
python notebooks/gain/compute_gain.py
```

### Outputs (saved to `notebooks/gain/`)

| File                  | Description               |
| --------------------- | ------------------------- |
| `gain_summary.csv`    | Gain values for all heads |
| `gain_fit_L*_H*.png`  | Individual head fit plots |
| `gain_comparison.png` | Bar chart comparing heads |

### Interpretation

| Gain Ratio (Target/Control) | Meaning                                         |
| --------------------------- | ----------------------------------------------- |
| ≈ 1.0                       | No functional distinction — universal mechanism |
| ≥ 1.5x                      | Target shows higher sensitivity                 |
| ≥ 2.0x                      | Strong evidence of head-specific exploitation   |
