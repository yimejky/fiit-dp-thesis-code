import logging

import torch
from torch import nn

""" source https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py """


def double_conv(in_channels, mid_out_channels, out_channels, kernel_size=3, padding=(1, 1, 1)):
    return nn.Sequential(
        nn.Conv3d(in_channels, mid_out_channels, kernel_size, padding=padding),
        nn.BatchNorm3d(mid_out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(mid_out_channels, out_channels, kernel_size, padding=(1, 1, 1)),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )


def down_double_conv(in_channels, out_channels, kernel_size=3, padding=(1, 1, 1)):
    return double_conv(in_channels, in_channels, out_channels, kernel_size, padding)


def up_double_conv(in_channels, out_channels, kernel_size=3, padding=(1, 1, 1)):
    return double_conv(in_channels, out_channels, out_channels, kernel_size, padding)


class UNetV2(nn.Module):
    """ source https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py """

    def __init__(self, in_channels=16, dropout_rate=0):
        super().__init__()
        out_channels = in_channels

        # downconv1 1=>x, x=>2x
        self.dconv_down1 = nn.Sequential(
            double_conv(1, out_channels, out_channels * 2),
            nn.Dropout(dropout_rate))
        out_channels *= 2
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/2

        # downconv2 2x=>2x, 2x=>4x
        self.dconv_down2 = nn.Sequential(
            down_double_conv(out_channels, out_channels * 2),
            nn.Dropout(dropout_rate))
        out_channels *= 2
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/4

        # downconv3 4x=>4x, 4x=>8x
        self.dconv_down3 = nn.Sequential(
            down_double_conv(out_channels, out_channels * 2),
            nn.Dropout(dropout_rate))
        out_channels *= 2
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/8

        # middle 8x=>8x, 8x=>16x
        self.dconv_middle = nn.Sequential(
            down_double_conv(out_channels, out_channels * 2),
            nn.Dropout(dropout_rate))
        out_channels *= 2
        logging.debug(f'UNet Architecture v2: max output channels {out_channels}')

        # upconv1 16x+8x=>8x, 8x=>8x
        self.up1 = nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2, bias=False)
        self.dconv_up1 = up_double_conv(out_channels + out_channels // 2, out_channels // 2)
        out_channels = out_channels // 2

        # upconv2 8x+4x=>4x, 4x=>4x
        self.up2 = nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2, bias=False)
        self.dconv_up2 = up_double_conv(out_channels + out_channels // 2, out_channels // 2)
        out_channels = out_channels // 2

        # upconv3 4x+2x=>2x, 2x=>2x
        self.up3 = nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2, bias=False)
        self.dconv_up3 = up_double_conv(out_channels + out_channels // 2, out_channels // 2)
        out_channels = out_channels // 2

        # final conv 2x=>1
        self.final = nn.Conv3d(out_channels, 1, kernel_size=1)
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

        h = self.dconv_middle(h)

        # up
        h = self.up1(h)
        h = torch.cat((h, d3), dim=1)
        h = self.dconv_up1(h)
        # print("after up1: ", h.data[0].shape)

        h = self.up2(h)
        h = torch.cat((h, d2), dim=1)
        h = self.dconv_up2(h)
        # print("after up2: ", h.data[0].shape)

        h = self.up3(h)
        h = torch.cat((h, d1), dim=1)
        h = self.dconv_up3(h)
        # print("after up3: ", h.data[0].shape)

        h = self.final(h)
        h = self.sigmoid(h)

        return h
