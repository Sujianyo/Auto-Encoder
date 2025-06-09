import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F  
# from utils.eval import dice_pytorch
class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.cross = nn.BCEWithLogitsLoss()
        # self.dice = DiceLoss()
        self.bce = BCE_Dice()
    def forward(self, output, label):
        return self.bce(output, label)
        # print(self.cross(output, label.unsqueeze(1)))
        # return self.cross(output, label.unsqueeze(1))
class BCE_Dice(nn.Module):
    def __init__(self, weight=None, size_average=True, alpha=0.01):
        super(BCE_Dice, self).__init__()
        self.alpha = alpha
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, output, target):
        bce = self.bce(output, target.unsqueeze(1))
        
        # return bce + self.alpha * self.dice(output, target).mean()
        return bce
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
