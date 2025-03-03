from model.unet_model import UNet
from dataset.drive import DRIVE_Dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm
from typing import Iterable
def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module, device: torch.device, epoch: int
        ):
    model.train()
    criterion.train()
    tbar = tqdm(data_loader)
    for idx, (image, label, _) in enumerate(tbar):
        outputs = model(image.to(device))
        # compute loss
        losses = criterion(outputs, label.to(device))
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        tbar.update(1)
        torch.cuda.empty_cache()
    return outputs, losses
