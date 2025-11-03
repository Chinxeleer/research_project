#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
model_name=Mamba
for pred_len in 1 7 14 30; do

	python -u run.py \
		--task_name long_term_forecast \
		--is_training 1 \
		--root_path ./dataset/exchange_rate/ \
		--data_path exchange_rate.csv \
		--model_id Exchange_$pred_len \
		--model $model_name \
		--data custom \
		--features M \
		--seq_len 96 \
		--label_len 48 \
		--pred_len $pred_len \
		--e_layers 2 \
		--d_layers 1 \
		--enc_in 5 \
		--expand 2 \
		--d_ff 64 \
		--dec_in 5 \
		--d_conv 4 \
		--c_out 5 \
		--d_model 128 \
		--des 'Exp' \
		--train_epochs 150 \
		--patience 999 \
		--itr 1

done
