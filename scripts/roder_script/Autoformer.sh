# export CUDA_VISIBLE_DEVICES=6

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path dataset_0315_2.csv \
  --model_id order_36_24 \
  --model Autoformer \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 24 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --use_gpu False
  ----target coeder