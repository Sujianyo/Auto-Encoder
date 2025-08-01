import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, active='relu', mem=False):
        super().__init__()
        if active == 'relu':
            act = nn.ReLU
        elif active == 'elu':
            act = nn.ELU
        if mem: 
            if in_channels < out_channels:
                self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1), 
                    nn.BatchNorm2d(in_channels),
                    act(inplace=True),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                    nn.BatchNorm2d(out_channels),
                    act(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    act(inplace=True)
                )  
            else:
                self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0), 
                    nn.BatchNorm2d(in_channels),
                    act(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    act(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    act(inplace=True)
                )  
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                act(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                act(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

# from .attention import *
from .swintrans import *
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, transformer=False, img_size=None, patch_size=None, window_size=None, heads=8):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)
        self.transformer = transformer
        if transformer:
            self.trans = SwinTransformer(img_size=img_size, patch_size=patch_size, in_chans=in_channels//2, window_size=window_size, num_heads=heads)
        # self.attention_layer = attention_layer
        # if attention_layer != 0:
        #     self.pos = PositionEncodingSine1DRelative(in_channels//2)
        #     self.trans = Transformer(in_channels//2, nhead=heads, num_attn_layers=attention_layer)
        # self.atten = []
        # self.atten_layer = attention_layer
        # if axial:
        #     self.self_atten = nn.ModuleList([
        #         Axial_Attention(out_channels, heads=heads, dropout=0.)
        #         for _ in range(attention_layer)
        #     ])

        # else:
        #     self.self_atten = nn.ModuleList([
        #         MultiHeadAttention(out_channels, heads=heads, dropout=0.)
        #         for _ in range(attention_layer)
        #     ])
        # self.self_atten = nn.ModuleList([
        #     MultiHeadAttention(in_channels//2, heads=heads, dropout=0.)
        #     for _ in range(attention_layer)
        # ])
        # self.cross_atten = nn.ModuleList([
        #     MultiHeadAttention(in_channels//2, heads=heads, dropout=0.)
        #     for _ in range(attention_layer)
        # ])

    def forward(self, x1, x2):
        
        x1 = self.up(x1)

        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # x = torch.cat([x2, x1], dim=1)
        # x = self.conv(x)
        # for i in range(self.atten_layer):
        #     x = cp.checkpoint(self.self_atten[i], x)
        # for i in range(self.atten_layer):
        # for layer1, layer2 in map(self.self_atten, self.cross_atten):
            # x1 = cp.checkpoint(self.self_atten[i], x1)
            # x2 = cp.checkpoint(self.cross_atten[i], x2, x1)
            # x1 = self.self_atten[i](x1)
            # x2 = self.cross_atten[i](x2, key=x1)
        # x = torch.cat([x1, x2], dim=1)
        if self.transformer != 0:
            x = self.trans(x2, x1)
            x = torch.cat([x, x1], dim=1)
        else:
            x = torch.cat([x2, x1], dim=1)
        # print(x.shape)
        return self.conv(x)




