import nibabel as nib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset.braTS import BraTSDataset

import os
import random
import numpy as np
import nibabel as nib
import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

def show_case(case_id, writer=None, global_step=0):
    path = os.path.join(TRAIN_DIR, case_id)
    flair = nib.load(os.path.join(path, f"{case_id}_flair.nii")).get_fdata()
    t1 = nib.load(os.path.join(path, f"{case_id}_t1.nii")).get_fdata()
    t1ce = nib.load(os.path.join(path, f"{case_id}_t1ce.nii")).get_fdata()
    t2 = nib.load(os.path.join(path, f"{case_id}_t2.nii")).get_fdata()
    seg = nib.load(os.path.join(path, f"{case_id}_seg.nii")).get_fdata()

    # Select middle slice
    slice_idx = flair.shape[2] // 2

    # Stack into shape [5, H, W]
    slices = np.stack([
        flair[:, :, slice_idx],
        t1[:, :, slice_idx],
        t1ce[:, :, slice_idx],
        t2[:, :, slice_idx],
        seg[:, :, slice_idx]
    ], axis=0)

    # Normalize for display
    slices = (slices - slices.min()) / (slices.max() - slices.min() + 1e-8)

    # Convert to tensor
    slices_tensor = torch.tensor(slices, dtype=torch.float32)  # [5, H, W]

    # Add channel dim -> [5, 1, H, W]
    slices_tensor = slices_tensor.unsqueeze(1)

    # Make grid -> [3, H, W] if you want 3-channel; for gray images, it's [1, H, W]
    img_grid = vutils.make_grid(slices_tensor, nrow=5, normalize=False, scale_each=False)

    # Write to TensorBoard
    if writer:
        writer.add_image(f"Case/{case_id}", img_grid, global_step)





image_size = (128, 128)
albumentations_transform = A.Compose([
    A.Resize(height=image_size[0], width=image_size[1]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
    A.Normalize(mean=0.0, std=1.0),  
    ToTensorV2()
])


# Path
TRAIN_DIR = "../brats20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

in_channel = 4
out_channel = 1
# attention_layer = 2
batch_size = 128
device = 'cuda:1'
labels = (1, 2, 4)
mode = "merged"

# logs = "./runs/brats/Transformer"
# from model.unet_model import UNet
# model = UNet(in_channel=in_channel, out_channel=out_channel, transformer=True, img_size=[16, 32, 64, 128], patch_size=[4, 4, 4, 4], window_size=[8, 8, 8, 8], heads=2).to(device)
# model_name='Unet_swin.pth'
# # model.load_state_dict(torch.load(model_name, weights_only=True))
# start_epoch = 0


logs = "./runs/brats/BaseLine"
from model.unet_model import UNet
model = UNet(in_channel=in_channel, out_channel=out_channel, transformer=False).to(device)
model_name='Unet_base.pth'
# model.load_state_dict(torch.load("Unet_base.pt", map_location=device))
# model.load_state_dict(torch.load(model_name))
start_epoch = 0


# Get samples
train_cases = sorted([d for d in os.listdir(TRAIN_DIR) if d.startswith("BraTS20_Training")])
sample_cases = random.sample(train_cases, 5)



train_dataset = BraTSDataset(TRAIN_DIR, filter_empty_mask=True, transforms=albumentations_transform, label_mode=mode, selected_labels=labels)






# TensorBoard writer
writer = SummaryWriter(log_dir=logs)

# Visualize
for i, case_id in enumerate(sample_cases):
    show_case(case_id, writer=writer, global_step=i)



# train_dataset = BraTSSliceDataset(
#     root_dir="/mnt/e/learning/dataset/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData",
#     name_map_csv="/mnt/e/learning/dataset/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/name_mapping.csv",
#     survival_csv="/mnt/e/learning/dataset/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv"
# )
# def albumentations_wrapper(albu_transform):
#     def transform(image, mask):
#         # PyTorch Tensor → NumPy
#         image = image.numpy().transpose(1, 2, 0)  # [C, H, W] → [H, W, C]
#         mask = mask.numpy()

#         # Apply Albumentations
#         augmented = albu_transform(image=image, mask=mask)

#         # Back to Tensor
#         image_tensor = torch.from_numpy(augmented["image"].transpose(2, 0, 1)).float()  # [H, W, C] → [C, H, W]
#         mask_tensor = torch.from_numpy(augmented["mask"]).long()

#         return image_tensor, mask_tensor
#     return transform

TRAIN_DATASET_PATH = '../brats20/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
# TEST_DATASET_PATH = '/mnt/e/learning/dataset/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_ValidationData/'
from torch.utils.data import random_split

train_ratio = 0.8
val_ratio = 1 - train_ratio
# test_dataset = BraTSDataset(TEST_DATASET_PATH, filter_empty_mask=True)
total_len = len(train_dataset)
train_len = int(train_ratio*total_len)
val_len = total_len - train_len
train_sub, val_sub = random_split(train_dataset, [train_len, val_len])

print(f"Total training samples: {len(train_dataset)}")
# print(f"Total testing samples: {len(test_dataset)}")
# Example single sample
img, mask = train_dataset[0]
print(mask.unique())
print(img.shape)   # Should be: [4, 240, 240]
print(mask.shape)  # Should be: [240, 240]
from utils.tensorboard_utils import show_aug2
train_dataloader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dataloader = DataLoader(val_sub, batch_size=batch_size, pin_memory=True)
print(f'Train samples:{len(train_sub)}')
print(f'Val samples:{len(val_sub)}')
show_aug2(train_dataloader, writer=writer)
# from model.unet_model import UNet
# net = UNet(in_channel=3, out_channel=1, transformer=True, img_size=[16, 32, 64, 128], patch_size=[4, 4, 4, 4], window_size=[8, 8, 8, 8], heads=4)

# model = UNet(in_channel=in_channel, out_channel=out_channel, transformer=True, img_size=[16, 32, 64, 128], patch_size=[4, 4, 4, 4], window_size=[8, 8, 8, 8], heads=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
from model.loss import Criterion
criterion = Criterion()
prev_best = np.inf

from utils.train import train_one_epoch, pre_heated
from utils.eval import evaluate

pre_heated(model, criterion, train_dataloader, device)

print("Start training")
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
        torch.save(model.state_dict(), model_name)
        prev_best = eval_stats['crs']
    # print('VAL:', eval_stats)
    writer.add_scalar("Loss/val", eval_stats['crs'], epoch)
    writer.add_scalar("IOU/val", eval_stats['iou'], epoch)
    writer.add_scalar("DICE/val", eval_stats['dice'], epoch)


writer.close()