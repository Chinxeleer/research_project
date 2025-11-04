#!/bin/bash

##############################################################################
# Mamba Model Training Script - Eden's Methodology
##############################################################################
# State Space Model for Financial Forecasting
# Optimized for stock price prediction using percentage change target
##############################################################################

# Model-specific parameters for Mamba
# CRITICAL: d_inner = d_model × expand MUST be <= 256
MODEL="Mamba"
D_MODEL=64          # Hidden dimension (64 × 2 = 128 ✓)
D_FF=16             # SSM state dimension (d_state)
D_CONV=4            # Convolution kernel size
EXPAND=2            # Expansion factor
E_LAYERS=2          # Number of encoder layers
DROPOUT=0.1

# Common parameters
SEQ_LEN=60
LABEL_LEN=30
BATCH_SIZE=32
LEARNING_RATE=0.0001
TRAIN_EPOCHS=100  # Increased for more thorough training
PATIENCE=10       # Increased from 5 to allow more exploration

ROOT_PATH="../dataset/processed_data/"
FEATURES="M"   # Multivariate (to match Autoformer)
TARGET="pct_chg"
ENC_IN=6
DEC_IN=6
C_OUT=6        # Multivariate output (to match Autoformer)

# Datasets and horizons
DATASETS=("NVIDIA" "APPLE" "SP500" "NASDAQ" "ABSA" "SASOL")
HORIZONS=(3 5 10 22 50 100)

echo "##############################################################################"
echo "# Training Mamba Model on All Datasets"
echo "##############################################################################"

for dataset in "${DATASETS[@]}"; do
    for horizon in "${HORIZONS[@]}"; do
        echo "Training: Mamba on $dataset for H=$horizon"

        python run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $ROOT_PATH \
            --data_path "${dataset}_normalized.csv" \
            --model_id "Mamba_${dataset}_H${horizon}" \
            --model $MODEL \
            --data custom \
            --features $FEATURES \
            --target $TARGET \
            --seq_len $SEQ_LEN \
            --label_len $LABEL_LEN \
            --pred_len $horizon \
            --enc_in $ENC_IN \
            --dec_in $DEC_IN \
            --c_out $C_OUT \
            --d_model $D_MODEL \
            --d_ff $D_FF \
            --d_conv $D_CONV \
            --expand $EXPAND \
            --e_layers $E_LAYERS \
            --dropout $DROPOUT \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --train_epochs $TRAIN_EPOCHS \
            --patience $PATIENCE \
            --use_gpu 1 \
            --gpu 0 \
            --use_wandb \
            --wandb_project "financial-forecasting-${dataset}" \
            --des 'Mamba_Exp' \
            --itr 1

        echo "Completed: $dataset H=$horizon"
        echo ""
    done
done

echo "Mamba training completed for all datasets!"
