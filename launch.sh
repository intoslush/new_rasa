#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
DATASET_NAME="CUHK-PEDES"

torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29502 \
  train.py \
  --name new_rasa \
  --checkpoint ./data/ALBEF/ALBEF.pth \
  --dataset_name $DATASET_NAME \
  --root_dir ./re_id \
  --num_epoch 5
  # --config configs/PS_cuhk_pedes.yaml \
