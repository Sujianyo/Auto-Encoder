
#!/usr/bin/env bash
python main_lgg.py  --epochs 200\
                --image_size 256\
                --device 'cuda:0'\
                --batch_size 8\
                --checkpoint pretrain\
                --pre_train\
                --num_workers 4\
                --dataset lgg_brain\
                --num_layer 4\
                --dataset_directory /home/yutong.cheng\