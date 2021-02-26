import logging

import torch
import numpy as np
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


def attention_block(in_channels, out_channels=1, kernel_size=1, padding=(0, 0, 0)):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.Sigmoid(),
    )


def down_double_conv(in_channels, out_channels, kernel_size=3, padding=(1, 1, 1)):
    return double_conv(in_channels, in_channels, out_channels, kernel_size, padding)


def up_double_conv(in_channels, out_channels, kernel_size=3, padding=(1, 1, 1)):
    return double_conv(in_channels, out_channels, out_channels, kernel_size, padding)


class UNetV3(nn.Module):
    """ source https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py """

    def __init__(self, in_channels=16,
                 input_data_channels=1,
                 output_label_channels=1,
                 dropout_rate=0):
        super().__init__()

        self.actual_epoch = 0
        self.actual_step = 0

        out_channels = in_channels


        # downconv1 y=>x, x=>2x
        self.dconv_down1 = nn.Sequential(
            double_conv(input_data_channels, out_channels, out_channels * 2),
            nn.Dropout(dropout_rate))
        self.dconv_atten1 = attention_block(in_channels=out_channels * 2)

        out_channels *= 2
        self.pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/2
        self.atten_pool1 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/2


        # downconv2 2x=>2x, 2x=>4x
        self.dconv_down2 = nn.Sequential(
            down_double_conv(out_channels, out_channels * 2),
            nn.Dropout(dropout_rate))
        self.dconv_atten2 = attention_block(in_channels=out_channels * 2)
        self.dconv_merge_atten2 = attention_block(in_channels=2)

        out_channels *= 2
        self.pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/4
        self.atten_pool2 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/4


        # downconv3 4x=>4x, 4x=>8x
        self.dconv_down3 = nn.Sequential(
            down_double_conv(out_channels, out_channels * 2),
            nn.Dropout(dropout_rate))
        self.dconv_atten3 = attention_block(in_channels=out_channels * 2)
        self.dconv_merge_atten3 = attention_block(in_channels=2)

        out_channels *= 2
        self.pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/8
        self.atten_pool3 = nn.MaxPool3d(2, stride=2, ceil_mode=True)  # 1/4


        # middle 8x=>8x, 8x=>16x
        self.dconv_middle = nn.Sequential(
            down_double_conv(out_channels, out_channels * 2),
            nn.Dropout(dropout_rate))
        self.middle_atten = attention_block(in_channels=out_channels * 2)
        self.middle_merge_atten = attention_block(in_channels=2)

        out_channels *= 2
        logging.debug(f'UNet Architecture v2: max output channels {out_channels}')


        # upconv1 16x+8x=>8x, 8x=>8x
        self.up1 = nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2, bias=False)
        self.dconv_up1 = up_double_conv(out_channels + out_channels // 2, out_channels // 2)

        self.atten_up1 = nn.ConvTranspose3d(1, 1, 2, stride=2, bias=False)
        self.atten_dconv_up1 = attention_block(in_channels=out_channels // 2)
        self.atten_dconv_merge_up1 = attention_block(in_channels=2)

        out_channels = out_channels // 2


        # upconv2 8x+4x=>4x, 4x=>4x
        self.up2 = nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2, bias=False)
        self.dconv_up2 = up_double_conv(out_channels + out_channels // 2, out_channels // 2)

        self.atten_up2 = nn.ConvTranspose3d(1, 1, 2, stride=2, bias=False)
        self.atten_dconv_up2 = attention_block(in_channels=out_channels // 2)
        self.atten_dconv_merge_up2 = attention_block(in_channels=2)

        out_channels = out_channels // 2


        # upconv3 4x+2x=>2x, 2x=>2x
        self.up3 = nn.ConvTranspose3d(out_channels, out_channels, 2, stride=2, bias=False)
        self.dconv_up3 = up_double_conv(out_channels + out_channels // 2, out_channels // 2)

        self.atten_up3 = nn.ConvTranspose3d(1, 1, 2, stride=2, bias=False)
        self.atten_dconv_up3 = attention_block(in_channels=out_channels // 2)
        self.atten_dconv_merge_up3 = attention_block(in_channels=2)

        out_channels = out_channels // 2


        # final conv 2x=>z
        self.final = nn.Conv3d(out_channels, output_label_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = x
        # print("init: ", x.data[0].shape)
        
        # DOWN1
        h = d1 = self.dconv_down1(h)
        p1 = self.dconv_atten1(d1)
        # print("inside pool2: ", h.data[0].shape, p1.data[0].shape, torch.matmul(h, p1).data[0].shape)
        h = self.pool1(torch.matmul(h, p1))
        p1 = self.atten_pool1(p1)
        # print("after pool1: ", h.data[0].shape)

        # DOWN2
        h = d2 = self.dconv_down2(h)
        p2 = self.dconv_atten2(d2)
        p2 = torch.cat((p1, p2), dim=1)
        p2 = self.dconv_merge_atten2(p2)
        # print("inside pool2: ", h.data[0].shape, p2.data[0].shape, torch.matmul(h, p2).data[0].shape)
        h = self.pool2(torch.matmul(h, p2))
        p2 = self.atten_pool2(p2)
        # print("after pool2: ", h.data[0].shape)

        # DOWN3
        h = d3 = self.dconv_down3(h)
        p3 = self.dconv_atten3(d3)
        p3 = torch.cat((p2, p3), dim=1)
        p3 = self.dconv_merge_atten3(p3)
        # print("inside pool3: ", h.data[0].shape, p3.data[0].shape, torch.matmul(h, p3).data[0].shape)
        h = self.pool3(torch.matmul(h, p3))
        p3 = self.atten_pool3(p3)
        # print("after pool3: ", h.data[0].shape)

        # MIDDLE
        h = self.dconv_middle(h)
        pm = self.middle_atten(h)
        pm = torch.cat((p3, pm), dim=1)
        pm = self.middle_merge_atten(pm)
        # print("middle: ", h.data[0].shape)

        # UP1
        # print("before up1: ", h.data[0].shape, pm.data[0].shape, torch.matmul(h, pm).data[0].shape)
        h = self.up1(torch.matmul(h, pm))
        ap1 = self.atten_up1(pm)

        h = torch.cat((h, d3), dim=1)
        h = self.dconv_up1(h)

        tmp_ap1 = self.atten_dconv_up1(h)
        ap1 = torch.cat((ap1, tmp_ap1), dim=1)
        ap1 = self.atten_dconv_merge_up1(ap1)
        # print("after up1: ", h.data[0].shape)

        # UP2
        # print("before up2: ", h.data[0].shape, ap1.data[0].shape, torch.matmul(h, ap1).data[0].shape)
        h = self.up2(torch.matmul(h, ap1))
        ap2 = self.atten_up2(ap1)

        h = torch.cat((h, d2), dim=1)
        h = self.dconv_up2(h)

        tmp_ap2 = self.atten_dconv_up2(h)
        ap2 = torch.cat((ap2, tmp_ap2), dim=1)
        ap2 = self.atten_dconv_merge_up2(ap2)
        # print("after up2: ", h.data[0].shape)

        # UP3
        # print("before up3: ", h.data[0].shape, ap2.data[0].shape, torch.matmul(h, ap2).data[0].shape)
        h = self.up3(torch.matmul(h, ap2))
        ap3 = self.atten_up2(ap2)

        h = torch.cat((h, d1), dim=1)
        h = self.dconv_up3(h)

        tmp_ap3 = self.atten_dconv_up3(h)
        ap3 = torch.cat((ap3, tmp_ap3), dim=1)
        ap3 = self.atten_dconv_merge_up3(ap3)
        # print("after up3: ", ap3.data[0].shape, h.data[0].shape)

        # FINAL
        # print("before final: ", h.data[0].shape, ap3.data[0].shape, torch.matmul(h, ap3).data[0].shape)
        h = self.final(torch.matmul(h, ap3))
        h = self.sigmoid(h)

        # print("DEBUG EPOCH AND STEP", self.actual_epoch, self.actual_step)
        if self.tensorboard_writer and self.actual_step == 0:
            single_index = 60
            tmp = ap3.data[0].detach().cpu().numpy()
            tmp = tmp.transpose(1, 0, 2, 3)
            # print('ATTENTION LAYER', np.min(tmp), np.max(tmp), tmp.shape)
            self.tensorboard_writer.add_image('single_sam_img', tmp[single_index], self.actual_epoch, dataformats="CHW")
            self.tensorboard_writer.add_images('batch_sam_img', tmp, self.actual_epoch, dataformats="NCHW")

            tmp = h.data[0].detach().cpu().numpy()
            tmp = tmp.transpose(1, 0, 2, 3)
            # print('OUTPUT', np.min(tmp), np.max(tmp), tmp.shape)
            self.tensorboard_writer.add_image('single_output_img', tmp[single_index], self.actual_epoch, dataformats="CHW")
            self.tensorboard_writer.add_images('batch_output_img', tmp, self.actual_epoch, dataformats="NCHW")

        return h
