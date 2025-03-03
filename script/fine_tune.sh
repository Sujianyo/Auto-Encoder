#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --epochs 800\
                --batch_size 2\
                --checkpoint DRIVE\
                --dataset DRIVE\
                --dataset_directory /mnt/c/dataset\
                # --resume /path