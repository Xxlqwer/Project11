import torch.nn as nn
import torch.nn.functional as F


class DownUp(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.stride) # 使用双线性插值函数进行上采样或下采样

class ResBlock(nn.ModuleList):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels), # 批归一化
            nn.LeakyReLU(), # LeakyReLU 激活函数
            nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.res = nn.Conv2d(in_channels, out_channels, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out) + self.res(x)
        return out

class ResNet(nn.ModuleList):
    def __init__(self):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            DownUp(0.5)
        )

        self.net = nn.ModuleList([
            nn.Sequential(
                ResBlock(32 * 2 ** i, 32 * 2 ** (i + 1)),
                ResBlock(32 * 2 ** (i + 1), 32 * 2 ** (i + 1)),
                DownUp(0.5)
            ) for i in range(4)
        ])

        self.out_layer = nn.Sequential(
            DownUp(0.125),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.Dropout(),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        out = self.in_layer(x)
        for i in range(len(self.net)):
            out = self.net[i](out)
        for i in range(len(self.out_layer)):
            out = self.out_layer[i](out)
        return out
