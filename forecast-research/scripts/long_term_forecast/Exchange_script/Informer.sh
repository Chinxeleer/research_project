#!/bin/sh
export CUDA_VISIBLE_DEVICES=0

for pred_len in 1 7 14 30; do

	python -u run.py \
		--task_name long_term_forecast \
		--is_training 1 \
		--root_path ./dataset/exchange_rate/ \
		--data_path exchange_rate.csv \
		--model_id exchange \
		--model Informer \
		--data custom \
		--features M \
		--seq_len 96 \
		--label_len 5 \
		--pred_len $pred_len \
		--d_model 128 \
		--d_ff 16 \
		--e_layers 2 \
		--d_layer 1 \
		--enc_in 5 \
		--dec_in 5 \
		--c_out 5 \
		--train_epochs 50 \
		--patience 100
done
