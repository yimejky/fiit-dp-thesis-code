import torch
from torch import nn


def double_conv(in_channels, out_channels, kernel_size=3, padding=(1, 1, 1)):
    """ source https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py """
    return nn.Sequential(
        nn.Conv3d(in_channels, in_channels, kernel_size, padding=padding),
        nn.BatchNorm3d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels, out_channels, kernel_size, padding=(1, 1, 1)),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    """ source https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py """
    """ source https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py """

    def __init__(self, in_channels=16):
        super().__init__()
        # torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        out_channels = in_channels  # 16

        # downconv1
        self.dconv_down1 = double_conv(1, out_channels, padding=(1, 1, 1))  # 1=>1, 1=>16
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/2

        # downconv2
        self.dconv_down2 = double_conv(out_channels, out_channels * 2)  # 16=>16, 16=>32
        out_channels *= 2
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/4

        # downconv3
        self.dconv_down3 = double_conv(out_channels, out_channels * 2)  # 32=>32, 32=>64
        out_channels *= 2
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/8

        # downconv4
        self.dconv_down4 = double_conv(out_channels, out_channels * 2)  # 64=>64, 64=>128
        out_channels *= 2
        self.pool4 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/16

        # middle
        self.dconv_middle = double_conv(out_channels, out_channels * 2)  # 128=>128, 128=>256
        out_channels *= 2
        print(f'max output channels {out_channels}')

        # upconv1
        self.up1 = nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2, bias=False)
        self.dconv_up1 = double_conv(out_channels + out_channels // 2,
                                     out_channels // 2)  # 256+128=>256+128, 256+128=>128
        out_channels = out_channels // 2

        # upconv2
        self.up2 = nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2, bias=False)
        self.dconv_up2 = double_conv(out_channels + out_channels // 2, out_channels // 2)  # 128+64=>128+64, 128+64=>64
        out_channels = out_channels // 2

        # upconv3
        self.up3 = nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2, bias=False)
        self.dconv_up3 = double_conv(out_channels + out_channels // 2, out_channels // 2)  # 64+32=>64+32, 64+32=>32
        out_channels = out_channels // 2

        # upconv4
        self.up4 = nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2, bias=False)
        self.dconv_up4 = double_conv(out_channels + out_channels // 2, out_channels // 2)  # 32+16=>32+16, 32+16=>16
        out_channels = out_channels // 2

        self.final = nn.Conv3d(out_channels, 1, kernel_size=1)  # 16=>1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = x
        # print("init: ", x.data[0].shape)

        # down
        h = d1 = self.dconv_down1(h)
        h = self.pool1(h)
        # print("after pool1: ", h.data[0].shape)

        h = d2 = self.dconv_down2(h)
        h = self.pool2(h)
        # print("after pool2: ", h.data[0].shape)

        h = d3 = self.dconv_down3(h)
        h = self.pool3(h)
        # print("after pool3: ", h.data[0].shape)

        h = d4 = self.dconv_down4(h)
        h = self.pool4(h)
        # print("after pool4: ", h.data[0].shape)

        h = self.dconv_middle(h)

        # up
        h = self.up1(h)
        h = torch.cat((h, d4), dim=1)
        h = self.dconv_up1(h)
        # print("after up1: ", h.data[0].shape)

        h = self.up2(h)
        h = torch.cat((h, d3), dim=1)
        h = self.dconv_up2(h)
        # print("after up2: ", h.data[0].shape)

        h = self.up3(h)
        h = torch.cat((h, d2), dim=1)
        h = self.dconv_up3(h)
        # print("after up3: ", h.data[0].shape)

        h = self.up4(h)
        h = torch.cat((h, d1), dim=1)
        h = self.dconv_up4(h)
        # print("after up4: ", h.data[0].shape)

        h = self.final(h)
        h = self.sigmoid(h)

        return h
