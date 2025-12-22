# Gain Analysis — Path A

Quantifies how strongly each attention head responds to query magnitude scaling.

## Purpose

Answers the question:

> Is the target head merely affected by query scaling (like all heads), or does it exploit query magnitude with significantly higher sensitivity?

## Metric

**Gain = slope of (mean_max_attn vs log(scale))**

Higher gain = higher sensitivity to query magnitude.

## Usage

```bash
# Auto-discover intervention CSVs
python notebooks/gain/compute_gain.py

# Explicit CSVs
python notebooks/gain/compute_gain.py \
    --target-csv "results/intervention/q_scale_intervention_L12_H0_target_*.csv" \
    --control-csv "results/intervention/q_scale_intervention_L12_H15_control_*.csv"
```

## Outputs (saved here)

| File                   | Description               |
| ---------------------- | ------------------------- |
| `gain_summary.csv`     | Gain values for all heads |
| `gain_fit_L*_H*_*.png` | Individual head fit plots |
| `gain_comparison.png`  | Bar chart comparing heads |

## Interpretation

| Gain Ratio | Meaning                           |
| ---------- | --------------------------------- |
| ≈ 1.0      | No functional distinction         |
| ≥ 1.5      | Target shows higher sensitivity   |
| ≥ 2.0      | Strong evidence of specialization |
