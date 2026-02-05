#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
# export https_proxy="http://127.0.0.1:7890"
# export http_proxy="http://127.0.0.1:7890"

NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
DATASET_NAME="CUHK-PEDES"

torchrun \
  --nproc_per_node=${NUM_GPUS} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29505 \
  train.py \
  --name new_rasa \
  --checkpoint ./data/ALBEF/ALBEF.pth \
  --dataset_name $DATASET_NAME \
  --root_dir ./re_id \
  --num_epoch 35 \
  --cluster_id_mode "cluster" \
  --massage "用来消融DBSCAN的超参" \
#instance  / cluster/unique_noise \
  # --config configs/PS_cuhk_pedes.yaml \
