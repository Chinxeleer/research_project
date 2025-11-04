#!/bin/bash

##############################################################################
# Master Experiment Runner - Following Eden Modise's Methodology
##############################################################################
# This script runs all experiments for the financial forecasting research
# Models: Mamba, Informer, Autoformer, FEDformer, iTransformer
# Datasets: NVIDIA, APPLE, SP500, NASDAQ, ABSA, SASOL, DRD_GOLD, ANGLO_AMERICAN
# Horizons: H=3, 5, 10, 22, 50, 100 (matching Eden's paper)
##############################################################################

# Configuration based on Eden's paper
SEQ_LEN=60          # Lookback window (Eden used 60 days)
LABEL_LEN=30        # Decoder start tokens (typically seq_len/2)
BATCH_SIZE=32       # Batch size
LEARNING_RATE=0.0001
TRAIN_EPOCHS=50     # Training epochs
PATIENCE=5          # Early stopping patience

# Data configuration
ROOT_PATH="../dataset/processed_data/"
FEATURES="M"        # Multivariate forecasting
TARGET="pct_chg"    # Percentage change (Eden's indirect modeling approach)

# GPU configuration
USE_GPU=1
GPU_ID=0

# Models to test
MODELS=("Mamba" "Informer" "Autoformer" "FEDformer" "iTransformer")

# Datasets (stock names)
DATASETS=("NVIDIA" "APPLE" "SP500" "NASDAQ" "ABSA" "SASOL" "DRD_GOLD" "ANGLO_AMERICAN")

# Forecast horizons from Eden's paper
HORIZONS=(3 5 10 22 50 100)

# Feature dimensions (must match CSV columns minus date)
# For our data: Open, High, Low, Close, Volume, pct_chg = 6 features
ENC_IN=6
DEC_IN=6
C_OUT=6

##############################################################################
# Function to run a single experiment
##############################################################################
run_experiment() {
    local model=$1
    local dataset=$2
    local pred_len=$3

    local data_path="${dataset}_normalized.csv"
    local model_id="${model}_${dataset}_H${pred_len}"

    echo "=========================================="
    echo "Running: $model on $dataset for H=$pred_len"
    echo "=========================================="

    python run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $ROOT_PATH \
        --data_path $data_path \
        --model_id $model_id \
        --model $model \
        --data custom \
        --features $FEATURES \
        --seq_len $SEQ_LEN \
        --label_len $LABEL_LEN \
        --pred_len $pred_len \
        --enc_in $ENC_IN \
        --dec_in $DEC_IN \
        --c_out $C_OUT \
        --patience $PATIENCE \
        --use_gpu $USE_GPU \
        --gpu $GPU_ID \
        --des 'Exp' \
        --itr 1

    echo "Completed: $model on $dataset for H=$pred_len"
    echo ""
}

##############################################################################
# Main execution
##############################################################################
echo "##############################################################################"
echo "# Financial Forecasting Experiments - Eden's Methodology"
echo "# Total experiments: ${#MODELS[@]} models × ${#DATASETS[@]} datasets × ${#HORIZONS[@]} horizons"
echo "##############################################################################"
echo ""

# Calculate total experiments
total_experiments=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#HORIZONS[@]}))
current_experiment=0

# Loop through all combinations
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for horizon in "${HORIZONS[@]}"; do
            current_experiment=$((current_experiment + 1))
            echo "Progress: Experiment $current_experiment of $total_experiments"
            run_experiment "$model" "$dataset" "$horizon"
        done
    done
done

echo "##############################################################################"
echo "# All experiments completed!"
echo "##############################################################################"
echo ""
echo "Results saved to:"
echo "  - Checkpoints: ./checkpoints/"
echo "  - Predictions: ./results/"
echo "  - Summary: ./result_long_term_forecast.txt"
