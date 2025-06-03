import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F  
from utils.eval import dice_pytorch
class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.cross = nn.BCEWithLogitsLoss()

    def forward(self, output, label):
        return self.cross(output, label)
    
def BCE_dice(output, target, alpha=0.01):
    bce = torch.nn.functional.binary_cross_entropy(output, target)
    soft_dice = 1 - dice_pytorch(output, target).mean()
    return bce + alpha * soft_dice