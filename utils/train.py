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
    losses = 0
    i = 0
    for idx, (image, label) in enumerate(tbar):
        outputs = model(image.to(device))
        # compute loss
        loss = criterion(outputs, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tbar.update(1)
        torch.cuda.empty_cache()
        # print(losses)
        losses += loss
        i += 1
    return outputs, losses/i
def pre_heated(model, loss, data, device):
    torch.backends.cudnn.benchmark = False
    a,b = next(iter(data))
    losss = loss(model(a.to(device)), b.to(device))
    losss.backward()
    torch.cuda.synchronize()