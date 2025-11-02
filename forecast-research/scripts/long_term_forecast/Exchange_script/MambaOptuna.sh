#!/bin/sh
export CUDA_VISIBLE_DEVICES=0 # Sets the GPU to use (index 0)
model_name=Mamba

for pred_len in 1 7 14 30; do

	python -u optuna_tune.py \
		--task_name long_term_forecast \
		--is_training 1 \
		--root_path ./dataset/exchange_rate/ \
		--data_path exchange_rate.csv \
		--model_id Exchange_$pred_len \
		--model $model_name \
		--data custom \
		--study_name Mamba_Exchange_$pred_len \
		--n_trials 50 \
		--features M \
		--seq_len 96 \
		--label_len 5 \
		--pred_len $pred_len \
		--e_layers 2 \
		--d_layers 1 \
		--enc_in 5 \
		--dec_in 5 \
		--c_out 5 \
		--d_model 128 \
		--des 'Exp' \
		--itr 1 \
		--d_ff 16 \
		--d_conv 4

done
