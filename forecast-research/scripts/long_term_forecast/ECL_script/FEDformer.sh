export CUDA_VISIBLE_DEVICES=0

model_name=FEDformer

for pred_len in 7 14 30 60 96; do

	python -u run.py \
		--task_name long_term_forecast \
		--is_training 1 \
		--root_path ./dataset/exchange_rate/ \
		--data_path exchange_rate.csv \
		--model_id Exchange_96_${pred_len} \
		--model $model_name \
		--data custom \
		--features M \
		--seq_len 96 \
		--label_len 48 \
		--pred_len $pred_len \
		--e_layers 2 \
		--d_layers 1 \
		--factor 3 \
		--enc_in 5 \
		--dec_in 5 \
		--c_out 1 \
		--des 'Exp' \
		--itr 1 \
		--train_epochs 15

done
