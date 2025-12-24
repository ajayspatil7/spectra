#!/bin/bash
# run_llama_experiment.sh
# =======================
# Re-runs LLaMA-3-8B experiments to verify GQA fix symmetry.
# Data is assumed to be in data/ctx{N} (defaults).

set -e

echo "========================================================"
echo "    Spectra Phase1-M: LLaMA-3-8B Regression Test"
echo "========================================================"

# 1. Environment Safety
echo "[1/2] Checking environment..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib

# Reinstall torchvision if needed (safety check, though likely fixed)
# pip install --force-reinstall torchvision

pip install -r requirements.txt > /dev/null

# 2. Run Pipeline
# Optional: Backup previous results (since we are checking for symmetry change)
if [ -d "results/Meta-Llama-3-8B" ]; then
    BACKUP_DIR="results/Meta-Llama-3-8B_backup_$(date +%Y%m%d_%H%M%S)"
    echo "⚠️  Found existing results. Backing up to $BACKUP_DIR..."
    mv "results/Meta-Llama-3-8B" "$BACKUP_DIR"
fi

# config/experiment.yaml is already configured for LLaMA
echo "[2/2] Running Experiment Pipeline..."
python run_pipeline.py

echo "========================================================"
echo "✅ Experiment Complete!"
echo "Results saved in: results/Meta-Llama-3-8B/"
echo "========================================================"
