#!/bin/bash

##############################################################################
# Autoformer Model Training Script
##############################################################################
# Auto-Correlation mechanism for long-term forecasting
# Decomposition architecture with trend-cyclical components
##############################################################################

MODEL="Autoformer"
D_MODEL=512
N_HEADS=8
E_LAYERS=2
D_LAYERS=1
D_FF=2048
DROPOUT=0.1
MOVING_AVG=25       # Moving average window for decomposition
FACTOR=1

SEQ_LEN=60
LABEL_LEN=30
BATCH_SIZE=32
LEARNING_RATE=0.0001
TRAIN_EPOCHS=50
PATIENCE=5

ROOT_PATH="../dataset/processed_data/"
FEATURES="M"
TARGET="pct_chg"
ENC_IN=6
DEC_IN=6
C_OUT=6

DATASETS=("NVIDIA" "APPLE" "SP500" "NASDAQ" "ABSA" "SASOL")
HORIZONS=(3 5 10 22 50 100)

echo "##############################################################################"
echo "# Training Autoformer Model on All Datasets"
echo "##############################################################################"

for dataset in "${DATASETS[@]}"; do
    for horizon in "${HORIZONS[@]}"; do
        echo "Training: Autoformer on $dataset for H=$horizon"

        python run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $ROOT_PATH \
            --data_path "${dataset}_normalized.csv" \
            --model_id "Autoformer_${dataset}_H${horizon}" \
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
            --n_heads $N_HEADS \
            --e_layers $E_LAYERS \
            --d_layers $D_LAYERS \
            --d_ff $D_FF \
            --dropout $DROPOUT \
            --moving_avg $MOVING_AVG \
            --factor $FACTOR \
            --batch_size $BATCH_SIZE \
            --learning_rate $LEARNING_RATE \
            --train_epochs $TRAIN_EPOCHS \
            --patience $PATIENCE \
            --use_gpu 1 \
	    --use_wandb \
            --wandb_project "financial-forecasting-${dataset}" \
            --gpu 0 \
            --des 'Autoformer_Exp' \
            --itr 1

        echo "Completed: $dataset H=$horizon"
        echo ""
    done
done

echo "Autoformer training completed for all datasets!"
