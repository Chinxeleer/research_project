#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

for pred_len in 1 7 14 30; do

	python -u run_wandb.py \
		--task_name long_term_forecast \
		--is_training 1 \
		--root_path ./dataset/exchange_rate/ \
		--data_path exchange_rate.csv \
		--use_wandb \
		--wandb_project "financial-forecasting-tuning" \
		--study_name Autoformer_Exchange_$pred_len \
		--n_trials 50 \
	--model_id exchange \
		--model Informer \
		--data custom \
		--features M \
		--seq_len 96 \
		--label_len 48 \
		--pred_len $pred_len \
		--d_model 256 \
		--d_ff 16 \
		--e_layers 2 \
		--d_layer 1 \
		--enc_in 5 \
		--dec_in 5 \
		--c_out 5 \
		--train_epochs 150 \
		--patience 999
done
