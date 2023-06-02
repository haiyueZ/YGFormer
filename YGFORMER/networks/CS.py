import os
import torch
import torch.nn as nn
from networks.dcd import conv_dy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DCW(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_dy(in_channels, mid_channels, 3,1,1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            conv_dy(mid_channels, out_channels, 3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x
class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()
        self.Sam = SpatialAttentionModul(in_channel=in_channel)
        self.dw = DCW(in_channels=in_channel,out_channels=in_channel)
    def forward(self, x):
        x = self.dw(x)
        #x = self.Cam(x)
        x = self.Sam(x)
        x = x.flatten(2).transpose(1, 2)
        return x



class SpatialAttentionModul(nn.Module):
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        MaxPool = torch.max(x, dim=1).values
        AvgPool = torch.mean(x, dim=1)
        MaxPool = torch.unsqueeze(MaxPool, dim=1)
        AvgPool = torch.unsqueeze(AvgPool, dim=1)
        x_cat = torch.cat((MaxPool, AvgPool), dim=1)
        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)
        x = Ms * x

        return x


