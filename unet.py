""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNet"]


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dilation_rate=1, padding=1,
                 normalization='batchnorm', num_groups=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if normalization == 'batchnorm':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=padding, dilation=dilation_rate),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation_rate),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        elif normalization == 'groupnorm':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=padding, dilation=dilation_rate),
                nn.GroupNorm(num_groups=num_groups, num_channels=mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation_rate),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
                nn.ReLU(inplace=True)
            )
        elif normalization == 'no_norm':
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=padding, dilation=dilation_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation_rate),
                nn.ReLU(inplace=True)
            )
        else:
            raise NotImplementedError("This type of normalization: {}, was not implemented.".format(normalization))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dilation_rate=1, padding=1, normalization='batchnorm', num_groups=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dilation_rate=dilation_rate, padding=padding,
                       normalization=normalization, num_groups=num_groups)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, dilation_rate=1, normalization='batchnorm',
                 num_groups=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dilation_rate=dilation_rate,
                                   normalization=normalization, num_groups=num_groups)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dilation_rate=dilation_rate, normalization=normalization,
                                   num_groups=num_groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


""" Full assembly of the parts to form the complete network """

class UNet(nn.Module):
    def __init__(self, num_in_channels, num_classes, bilinear=False, dropout_probability=0, normalization='batchnorm',
                 num_groups=1):
        super(UNet, self).__init__()

        self.n_channels = num_in_channels
        self.n_classes = num_classes
        self.bilinear = bilinear

        self.normalization = normalization
        self.num_groups = num_groups

        self.inc = DoubleConv(self.n_channels, 64, normalization=self.normalization, num_groups=self.num_groups)
        self.down1 = Down(64, 128, normalization=self.normalization, num_groups=self.num_groups)
        self.down2 = Down(128, 256, normalization=self.normalization, num_groups=self.num_groups)
        self.down3 = Down(256, 512, normalization=self.normalization, num_groups=self.num_groups)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor, normalization=self.normalization, num_groups=self.num_groups)
        self.up1 = Up(1024, 512 // factor, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)
        self.up2 = Up(512, 256 // factor, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)
        self.up3 = Up(256, 128 // factor, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)
        self.up4 = Up(128, 64, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)
        self.outc = OutConv(64, self.n_classes)
        self.softmax = nn.Softmax2d()
        self.dropout = nn.Dropout2d(p=dropout_probability)

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
        x = self.dropout(x)
        logits = self.outc(x)
        # preds = self.softmax(logits)
        return logits


""" UNet with additional intermediate output-heads """

class HierarchicalUNet(nn.Module):
    def __init__(self, num_in_channels, num_classes, bilinear=False, dropout_probability=0.0, normalization='batchnorm',
                 num_groups=1):
        super(HierarchicalUNet, self).__init__()

        # Basic configuration parameters
        self.n_channels = num_in_channels
        self.n_classes = num_classes
        self.bilinear = bilinear
        self.normalization = normalization
        self.num_groups = num_groups
        self.dropout = nn.Dropout2d(p=dropout_probability)

        # Standard UNet layer definitions
        self.inc = DoubleConv(self.n_channels, 64, normalization=self.normalization, num_groups=self.num_groups)
        self.down1 = Down(64, 128, normalization=self.normalization, num_groups=self.num_groups)
        self.down2 = Down(128, 256, normalization=self.normalization, num_groups=self.num_groups)
        self.down3 = Down(256, 512, normalization=self.normalization, num_groups=self.num_groups)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor, normalization=self.normalization, num_groups=self.num_groups)
        self.up1 = Up(1024, 512 // factor, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)
        self.up2 = Up(512, 256 // factor, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)
        self.up3 = Up(256, 128 // factor, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)
        self.up4 = Up(128, 64, self.bilinear, normalization=self.normalization, num_groups=self.num_groups)

        # Output head on the outermost layer (full-scale output)
        self.outc_mask = torch.nn.Sequential(OutConv(64, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                             OutConv(64, self.n_classes))

        # Hierarchically integrated intermediate output-heads
        self.outc_x5 = torch.nn.Sequential(OutConv(1024 // factor, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                           OutConv(64, self.n_classes))
        self.outc_up1 = torch.nn.Sequential(OutConv(512 // factor, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                            OutConv(64, self.n_classes))
        self.outc_up2 = torch.nn.Sequential(OutConv(256 // factor, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                            OutConv(64, self.n_classes))
        self.outc_up3 = torch.nn.Sequential(OutConv(128 // factor, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                            OutConv(64, self.n_classes))
        self.outc_up4 = torch.nn.Sequential(OutConv(64, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                            OutConv(64, self.n_classes))

    def forward(self, x):
        # UNet encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # UNet decoder + intermediate output-heads
        x5_logit = self.outc_x5(x5)

        x = self.up1(x5, x4)
        up1_logit = self.outc_up1(x)

        x = self.up2(x, x3)
        up2_logit = self.outc_up2(x)

        x = self.up3(x, x2)
        up3_logit = self.outc_up3(x)

        x = self.up4(x, x1)
        up4_logit = self.outc_up4(x)

        x = self.dropout(x)

        # Full-scale output conv
        logit_mask_ce = self.outc_mask(x)

        return logit_mask_ce, [x5_logit, up1_logit, up2_logit, up3_logit, up4_logit]
