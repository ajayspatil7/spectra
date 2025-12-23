# EXP3b — Gain-Conditioned Q vs K Scaling

**Status:** 🔄 Ready to run

**FINAL EXPERIMENT IN SPECTRA**

## Scientific Question

> Does Q–K asymmetry in attention entropy emerge selectively in high-gain heads,
> or is attention always governed by generic logit magnitude?

## Why This Experiment Exists

- Exp3a showed Q ≈ K symmetry for a medium-gain head
- Exp2b showed strong heterogeneity of gain across heads
- Exp3b tests whether control regimes differ by gain class

This is **NOT exploratory**. This is a **CONFIRMATORY, STRATIFIED CONTROL EXPERIMENT**.

## Target Heads (LOCKED)

| Name            | Layer | Head | Expected Gain |
| --------------- | ----- | ---- | ------------- |
| **High-gain**   | 20    | 19   | ~0.804        |
| **Medium-gain** | 14    | 8    | ~0.50         |
| **Low-gain**    | 0     | 23   | ~0.017        |

⚠️ These heads are fixed. No cherry-picking after results.

## Run Command

```bash
python experiments/exp3b_gain_conditioned/run_exp3b.py
```

## Outputs

| File                         | Description              |
| ---------------------------- | ------------------------ |
| `exp3b_combined.png`         | All 3 heads side-by-side |
| `exp3b_summary.csv`          | Asymmetry metrics        |
| `L*_H*/qk_delta_entropy.png` | **PRIMARY FIGURES**      |
| `L*_H*/qk_results.csv`       | Per-head data            |

## Interpretation Rules (STRICT)

| Head Type   | If Q ≠ K               | If Q ≈ K                   |
| ----------- | ---------------------- | -------------------------- |
| High-gain   | Query-dominant control | Generic logit scaling      |
| Medium-gain | Transitional regime    | Generic regime             |
| Low-gain    | Unexpected             | ✅ Negative control passed |

## What This Experiment Can Claim

✅ Attention control regimes are **heterogeneous**
✅ Gain stratifies heads into **functional classes**
✅ Q–K asymmetry is **not universal**

## Stopping Condition

After EXP3b:

- 🚫 No more experiments
- 🚫 No more heads
- 🚫 No new metrics

**The experimental phase is over.**
