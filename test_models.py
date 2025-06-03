
from model.unet_model import UNet
from model.loss import Criterion
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm

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



device = torch.device('cuda')
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

import torch

torch.cuda.reset_peak_memory_stats()

model = UNet(attention_layer=1).to(device)
# traced = symbolic_trace(model.to('cpu'))

# writer = SummaryWriter("runs/detailed_model")
dummy_input = torch.randn(1, 3, 254, 254).to(device)
model(dummy_input)
# writer.add_graph(traced, dummy_input)
# writer.close()

max_mem = torch.cuda.max_memory_allocated()

print(f"Max memery: {max_mem / 1024**2:.2f} MB")


