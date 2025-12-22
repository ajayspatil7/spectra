# Spectra Phase 1 Checklist

**Objective**: Empirically verify whether Query Norm (‖Q‖) predicts Attention Entropy across layers and heads in Llama-3-8B.

---

## ✅ Completed

### Environment & Repository

- [x] **Initialize Git repository** — Created project structure, pushed to GitHub
- [x] **Define experiment config** — `src/config.py` with frozen hyperparameters (fp16, 4K context, batch=1)
- [x] **Requirements file** — `requirements.txt` with PyTorch, transformers, scipy, matplotlib

### Experiment Zero (Validation)

- [x] **Basic inference script** — `notebooks/experiment_zero/basic_inference.py`
- [x] **Validate on SageMaker** — Tesla T4, model loads (12.83 GB), inference works (1.24 tok/s)
- [x] **Model dissection script** — `notebooks/experiment_zero/dissect_model.py`
  - Architecture overview (32 layers, 32 heads, 8 KV heads, GQA)
  - Q/K/V projection visualization
  - Manual attention computation (step-by-step)
  - Query norm computation (‖Q‖₂)
  - Attention entropy computation (mask-aware, NaN-safe)
  - Per-head correlation demo

---

## 🔲 To Do

### Data Preparation

- [ ] **Prepare long-context input** — 4K tokens of diverse text
- [ ] **Create data loader** — `src/data_loader.py`

### Core Experiment

- [ ] **Implement attention hooks** — `src/hooks.py` to capture Q and attention probs across ALL 32 layers
- [ ] **Implement metrics module** — `src/metrics.py` with query_norm() and attention_entropy()
- [ ] **Build main experiment script** — `scripts/run_experiment.py`
- [ ] **Run full experiment** — Collect (layer, head, token, q_norm, entropy) tuples

### Analysis

- [ ] **Compute correlations** — Pearson + Spearman for each (layer, head) pair
- [ ] **Run randomization control** — Shuffle entropy, verify correlations → ~0
- [ ] **Save raw data** — CSV/pickle with all collected metrics

### Visualization

- [ ] **Scatter plots** — ‖Q‖ vs entropy for representative heads
- [ ] **Correlation heatmap** — Layers × Heads color-coded by r
- [ ] **Distribution histograms** — Query norm and entropy distributions

### Deliverables

- [ ] **Write interpretation** — Document findings, layer-by-layer patterns
- [ ] **Go/No-Go decision** — Based on |r| ≥ 0.5, p < 0.01 criteria
- [ ] **Final commit** — Tag as `phase1-complete`

---

## Success Criteria (Fixed Before Analysis)

| Metric                   | Threshold                           |
| ------------------------ | ----------------------------------- |
| Correlation magnitude    | \|r\| ≥ 0.5 in meaningful subset    |
| Statistical significance | p < 0.01                            |
| Randomization control    | Shuffled correlations → ~0          |
| Reproducibility          | Results hold across multiple inputs |

---

## File Structure (Target)

```
Spectra/
├── src/
│   ├── config.py        ✅ Done
│   ├── hooks.py         🔲 To Do
│   ├── metrics.py       🔲 To Do
│   └── data_loader.py   🔲 To Do
├── scripts/
│   ├── run_experiment.py    🔲 To Do
│   └── visualize.py         🔲 To Do
├── notebooks/
│   └── experiment_zero/
│       ├── basic_inference.py   ✅ Done
│       └── dissect_model.py     ✅ Done
├── results/                     🔲 To Do (experiment outputs)
├── CHECKLIST.md                 ✅ This file
└── README.md                    ✅ Done
```
