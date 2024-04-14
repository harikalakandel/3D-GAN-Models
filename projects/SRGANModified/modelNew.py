#https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/SRGAN/model.py
import torch
from torch import nn

#from pixelshuffle3d import *

from projects.SRGANModified.pixelshuffle3d import *



class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=True,
        use_bn=False,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv3d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm3d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.ReLU()
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor):#scale_factor=2 always
        super().__init__()
       
        self.conv = nn.Conv3d(in_c, in_c * scale_factor ** 2, 1, 1, 1)        
        self.ps = PixelShuffle3d(scale_factor)
        self.act = nn.ReLU()
        

    def forward(self, x):       
        return self.act(self.ps(self.conv(x)))



class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=True,#False,
        )
        self.block3 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=True,#False
        )
        self.block4 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=True,#False,
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        return out + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=4, num_blocks=4):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=True)
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 1))
        #return 1 channel to match with real mri
        self.final = nn.Conv3d(num_channels, 1, kernel_size=11, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial        
        x=self.upsamples(x)
        
        return torch.tanh(self.final(x))
      
    
    
class Generator3To1(nn.Module):
    def __init__(self, in_channels=3, num_channels=4, num_blocks=4):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=True)
        
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 1))
        #return 1 channel to match with real mri
        self.final = nn.Conv3d(num_channels, 1, kernel_size=11, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial        
        x=self.upsamples(x)
        
        return torch.tanh(self.final(x))
       


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[8,8,16,16,32,32,64,64]):#4,4,8,8,32,32,64, 64, 128, 128]):#, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        AVG_PULLING=32#16#8#6
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False #if idx == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((AVG_PULLING, AVG_PULLING, AVG_PULLING)),
            nn.Flatten(),
            nn.Linear(64*AVG_PULLING*AVG_PULLING*AVG_PULLING, 128),
            #nn.Linear(features[-1]*AVG_PULLING*AVG_PULLING*AVG_PULLING, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            #nn.Sigmoid()#nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return  self.classifier(x)
        
    
    
