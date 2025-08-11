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
        if label.ndim!=output.ndim:
            return self.bce(output.squeeze(1), label)
        return self.bce(output, label)

# class BCE_Dice(nn.Module):
#     def __init__(self, weight=None, size_average=True, alpha=0.01):
#         super(BCE_Dice, self).__init__()
#         self.alpha = alpha
#         self.dice = DiceLoss()
#         self.bce = nn.BCEWithLogitsLoss()

#     def forward(self, output, target):
#         if output.ndim != target.ndim:
#             target = target.unsqueeze(1)
#         bce = self.bce(output, target.float())
#         dice = self.dice(output, target)
#         return bce + self.alpha * dice
    
class BCE_Dice(nn.Module):
    def __init__(self, alpha=0.01):
        super(BCE_Dice, self).__init__()
        self.alpha = alpha
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, output, target):
        if output.ndim == 3:  # [B, H, W] → add channel dim
            output = output.unsqueeze(1)
            target = target.unsqueeze(1)
        bce = self.bce(output, target.float())
        dice = self.dice(output, target)
        return bce + self.alpha * dice

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, inputs, targets):
#         inputs = torch.sigmoid(inputs)  # safe version
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + self.smooth) / \
#                (inputs.sum() + targets.sum() + self.smooth)
#         return 1 - dice

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: [B, C, H, W] or [B, 1, H, W]
        targets: same shape as inputs
        """
        inputs = torch.sigmoid(inputs)  # safe version

        # flatten per channel
        B, C = inputs.shape[:2]
        inputs = inputs.view(B, C, -1)
        targets = targets.view(B, C, -1)

        intersection = (inputs * targets).sum(dim=2)
        dice = (2. * intersection + self.smooth) / \
               (inputs.sum(dim=2) + targets.sum(dim=2) + self.smooth)

        # return mean dice loss over batch & channels
        return 1 - dice.mean()
