import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()

        self.downscale = nn.Sequential(
            nn.Conv3d(3, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(out_channels * 2),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels * 2, out_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels * 2),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels * 2, out_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels * 2),
            nn.LeakyReLU(),
        )
        self.upscale = nn.Sequential(
            nn.ConvTranspose3d(out_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(out_channels, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img = self.downscale(img)
        img = self.upscale(img)

        return F.pad(img, pad=(0, 1, 0, 0, 0, 1))


class Discriminator(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        
        self.dis = nn.Sequential(
            nn.Conv3d(1, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels, out_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(out_channels * 2),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels * 2, out_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm3d(out_channels * 4),
            nn.LeakyReLU(),
            nn.Conv3d(out_channels * 4, 1, kernel_size=2, stride=1, padding=0),
        )
        
    def forward(self, img):
        out = self.dis(img)
        
        return out.squeeze()
