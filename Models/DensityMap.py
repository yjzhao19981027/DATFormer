import torch
import torch.nn as nn
import math


class DensityMap(nn.Module):
    def __init__(self):
        super(DensityMap, self).__init__()
        self.downsampling1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.downsampling2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )
        self.downsampling3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )

    def generateDensityMap(self, x):
        _, _, H, _ = x.shape
        density_map = torch.ones_like(x)
        for i in range(H):
            # if i < H / 4 or i > 3 * H / 4:
                # density_map[:, :, i, :] = 0
            theta = math.fabs(i - H // 2) / (H // 2) * math.pi / 2
            density_map[:, :, i, :] = math.cos(theta)
        return density_map

    def forward(self, x):
        density_map = self.generateDensityMap(x)
        density_map = self.downsampling1(density_map)
        density_map = self.downsampling2(density_map)
        density_map = self.downsampling3(density_map)

        return density_map