
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


model = UNet().to(device)
traced = symbolic_trace(model.to('cpu'))

writer = SummaryWriter("runs/detailed_model")
dummy_input = torch.randn(1, 3, 254, 254)
writer.add_graph(traced, dummy_input)
writer.close()
