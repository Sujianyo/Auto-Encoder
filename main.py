
from model.unet_model import UNet
from model.loss import Criterion
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm
import torch.utils.data as data

import os
import random
import numpy as np
import torch
from utils.train import train_one_epoch
from dataset import build_data_loader
from utils.eval import evaluate
from utils.inference import inference

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.fx import symbolic_trace
from dataset.brain_35h import Br35H
from datetime import datetime

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def log_final_predictions_to_tensorboard(model, dataloader, device, writer, num_images=4):
    import torch
    import torchvision
    model.eval()
    count = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)       # shape: (B, 3, H, W)
            targets = targets.to(device)     # shape: (B, 1, H, W) or (B, H, W)

            outputs = model(inputs)          # shape: (B, 1, H, W)

            for i in range(inputs.size(0)):
                if count >= num_images:
                    return

                input_img = inputs[i].cpu()       # (3, H, W)
                target_img = targets[i].cpu()     # (1, H, W) or (H, W)
                output_img = outputs[i].cpu()     # (1, H, W) or (H, W)

                if target_img.ndim == 3:
                    target_img = target_img.squeeze(0)
                if output_img.ndim == 3:
                    output_img = output_img.squeeze(0)

                input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
                target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min() + 1e-8)
                output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min() + 1e-8)

                target_color = torch.stack([target_img]*3, dim=0)  # (3, H, W)
                output_color = torch.stack([output_img]*3, dim=0)  # (3, H, W)

                grid = torch.cat([input_img, target_color, output_color], dim=2)  # (3, H, 3W)
                writer.add_image(f"Compare/{count}", grid, global_step=0)

                count += 1



device = torch.device('cuda')
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
experiment_name = 'unet_BR35'
dataset_dir = '/mnt/e/learning/Dataset'
batch_size = 1
log_dir = os.path.join("runs", experiment_name or datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=log_dir)
model = UNet(attention_layer=0).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


dataset_train = Br35H(dataset_dir, 'train')
dataset_val = Br35H(dataset_dir, 'val')
dataset_test = Br35H(dataset_dir, 'test')
data_loader_train = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
data_loader_val = data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
data_loader_test = data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
criterion = Criterion()
prev_best = np.inf

# train
print("Start training")
start_epoch = 0
epochs = 30
for epoch in range(start_epoch, epochs):
    # train
    print("Epoch: %d" % epoch)
    
    _, train_loss = train_one_epoch(model, data_loader_train, optimizer, criterion, device, epoch)
    writer.add_scalar("Loss/train", train_loss, epoch)


    torch.cuda.empty_cache()
    print("Start evaluation")
    eval_stats = evaluate(model, criterion, data_loader_train, device, epoch, False)
    print('VAL:', eval_stats)
    writer.add_scalar("Loss/val", eval_stats['crs'], epoch)
    writer.add_scalar("IOU/val", eval_stats['iou'], epoch)
    writer.add_scalar("IOU/val", eval_stats['dice'], epoch)
log_final_predictions_to_tensorboard(model, data_loader_val, device, writer, num_images=6)
writer.close()







