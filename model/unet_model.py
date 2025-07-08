import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_channel = 3, out_channel = 1, bilinear=True, attention_layer = 0, axial=False, active='relu', mem=False):
        super(UNet, self).__init__()
        self.n_channels = in_channel
        self.n_classes = out_channel
        self.bilinear = bilinear
        # Encoder 
        self.inc = DoubleConv(in_channel, 64, active=active, mem=mem)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        # Decoder
        self.up1 = Up(1024, 256, bilinear, attention_layer=attention_layer)
        self.up2 = Up(512, 128, bilinear, attention_layer=attention_layer)
        self.up3 = Up(256, 64, bilinear, attention_layer=attention_layer )
        self.up4 = Up(128, 64, bilinear, attention_layer=attention_layer)
        self.outc = OutConv(64, out_channel)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    print(net)