import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F  

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.cross = nn.BCEWithLogitsLoss()

    def forward(self, output, label):
        return self.cross(output, label)