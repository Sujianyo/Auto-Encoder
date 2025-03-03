#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0
python main.py  --batch_size 1\
                --inference \
                --dataset DRIVE\
                --dataset_directory /mnt/c/dataset\
                --resume /your/model/path