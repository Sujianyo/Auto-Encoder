
#!/usr/bin/env bash
python main.py  --epochs 200\
                --device 0\
                --batch_size 8\
                --checkpoint pretrain\
                --pre_train\
                --num_workers 4\
                --dataset lgg_brain\
                --nheads 8\
                --num_layers 4\
                --dataset_directory /home/yutong.cheng\