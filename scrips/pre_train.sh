
#!/usr/bin/env bash
python main_lgg.py  --epochs 200\
                --patchify\
                --image_size 128\
                --device 'cuda:1'\
                --batch_size 8\
                --checkpoint pretrain\
                --pre_train\
                --num_workers 4\
                --dataset lgg_brain\
                --num_layer 4\
                --dataset_directory /home/yutong.cheng