import torch.nn.functional as F

# from .unet_parts import *
from unet_parts import *

class UNet(nn.Module):
    def __init__(self, in_channel = 3, out_channel = 1, bilinear=True, transformer=False, img_size=[None, None, None, None], patch_size=[None, None, None, None], window_size=[None, None, None, None], heads=4, active='relu', mem=False):
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
        self.up1 = Up(1024, 256, bilinear, transformer=transformer, img_size=img_size[0], patch_size=patch_size[0], window_size=window_size[0], heads=heads)
        self.up2 = Up(512, 128, bilinear, transformer=transformer, img_size=img_size[1], patch_size=patch_size[1], window_size=window_size[1], heads=heads)
        self.up3 = Up(256, 64, bilinear, transformer=transformer, img_size=img_size[2], patch_size=patch_size[2], window_size=window_size[2], heads=heads)
        self.up4 = Up(128, 64, bilinear, transformer=transformer, img_size=img_size[3], patch_size=patch_size[3], window_size=window_size[3], heads=heads)
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
    net = UNet(in_channel=3, out_channel=1, transformer=True, img_size=[128, 64, 32, 16], patch_size=[4, 4, 4, 4], window_size=[8, 8, 8, 8], heads=4)
    # print(net)
    x = torch.rand(1, 3, 128, 128)
    net(x).shape