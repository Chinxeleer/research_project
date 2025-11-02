#!/bin/bash

# This script submits all forecasting model jobs via sbatch
# Usage: bash run_all_models.sh

BASE_DIR="./scripts/long_term_forecast/Exchange_script"

echo "=========================================="
echo " Submitting all model training jobs..."
echo "=========================================="

# Submit Mamba job
echo "Submitting Mamba model..."
sbatch "$BASE_DIR/Mamba_wandb.sh"

# Submit Autoformer job
echo "Submitting Autoformer model..."
sbatch "$BASE_DIR/Autoformer_wandb.sh"

# Submit Informer job
echo "Submitting Informer model..."
sbatch "$BASE_DIR/Informer_wandb.sh"

# Submit FEDformer job
echo "Submitting FEDformer model..."
sbatch "$BASE_DIR/FEDformer_wandb.sh"

echo "==========================================
