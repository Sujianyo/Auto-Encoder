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
from utils.tensorboard_utils import *
# from dataset import build_data_loader
from utils.eval import evaluate
from utils.inference import inference
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# from torch.fx import symbolic_trace
from dataset.brain_lgg import MRI_Dataset
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
from sklearn.model_selection import train_test_split

PATCH_SIZE = 128

transforms = A.Compose([
    A.Resize(width = PATCH_SIZE, height = PATCH_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
    A.Normalize(p=1.0),
    ToTensorV2(),
])

device = torch.device('cuda:1')
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
experiment_name = 'lgg_brain'


## MODEL Name
experiment_name = 'lgg_brain'
model_name = 'Unet_self_attention'
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("runs", experiment_name, model_name, timestamp)
writer = SummaryWriter(log_dir=log_dir)


mask_paths = glob.glob('/home/yutong.cheng/kaggle_3m/*/*_mask*')
image_paths = [i.replace('_mask', '') for i in mask_paths]
print(f'{len(mask_paths)} {len(image_paths)}')
print(f'Logs are saved to: {log_dir}')
# log_dir = os.path.join("runs", experiment_name or datetime.now().strftime("%Y%m%d-%H%M%S"))
# writer = SummaryWriter(log_dir=log_dir)
# print(plot_to_tensorboard )
plot_to_tensorboard(5, list_img_paths=image_paths[5:], list_mask_paths=mask_paths[5:], writer=writer, tag_name='Visualizing')

df = pd.DataFrame(data={'image': image_paths, 'mask': mask_paths})
df.head()
plot_to_tensorboard(n_examples=3, list_img_paths=df.iloc[[3, 4, 5], 0].tolist(), list_mask_paths=df.iloc[[3, 4, 5], 1].tolist(), writer=writer, tag_name='Dataframe')

df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
df_test, df_val = train_test_split(df_test, test_size=0.3, random_state=42)
print('train')
print(f"{df_train.describe().loc['count', ['image', 'mask']]}\n")
print('val')
print(f"{df_val.describe().loc['count', ['image', 'mask']]}\n")
print('test')
print(f"{df_test.describe().loc['count', ['image', 'mask']]}\n")


train_dataset = MRI_Dataset(df_train, img_transform=transforms)
val_dataset = MRI_Dataset(df_val, img_transform=transforms)
test_dataset = MRI_Dataset(df_test, img_transform=transforms)

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
show_aug(train_dataloader, writer)

for img, mask in train_dataloader:
    print(img.shape)
    print(mask.shape)
    break

model = UNet(attention_layer=1).to(device)
# model = torch.load("model.pt", map_location=device, weights_only=False)
# model.load_state_dict(torch.load("model.pt", map_location=device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
criterion = Criterion()
prev_best = np.inf

print("Start training")
start_epoch = 0
epochs = 200
for epoch in range(start_epoch, epochs):
    # train
    print("Epoch: %d" % epoch)
    
    _, train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device, epoch)
    print('train_loss', train_loss)
    writer.add_scalar("Loss/train", train_loss, epoch)


    torch.cuda.empty_cache()
    print("Start evaluation")
    eval_stats = evaluate(model, criterion, val_dataloader, device, epoch, False)
    if eval_stats['crs'] < prev_best:
        torch.save(model.state_dict(), 'model.pt')
        prev_best = eval_stats['crs']
    # print('VAL:', eval_stats)
    writer.add_scalar("Loss/val", eval_stats['crs'], epoch)
    writer.add_scalar("IOU/val", eval_stats['iou'], epoch)
    writer.add_scalar("DICE/val", eval_stats['dice'], epoch)

visualize_segmentation_tensorboard(model, test_dataloader, device, writer, num_examples=4)

writer.close()
