#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0  
python main.py  --epochs 40 \
                --batch_size 2 \
                --pre_train \
                --dataset DRIVE \
                --dataset_directory /mnt/c/dataset \
