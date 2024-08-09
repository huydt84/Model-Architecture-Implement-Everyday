import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .se_block import SEBlock2D

class SEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8, downsample=None, stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.downsample = downsample
        self.se_block = SEBlock2D(out_channels, reduction)   # add se block
        
    def forward(self, x):
        out = self.layer(x)
        out = self.se_block(out)   # calculate se output inside skip connection
        if self.downsample is not None:
            x = self.downsample(x)
        out = F.relu(out + x)
        return x
    
class SEResNet18(nn.Module):
    def __init__(self, image_channels=3, num_classes=1000, reduction=8):
        super().__init__()
        # Normal sequential convolution layer
        self.layer = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual connection layer
        self.res1 = self.__make_layer(64, 64, stride=1, reduction=reduction)
        self.res2 = self.__make_layer(64, 128, stride=2, reduction=reduction)
        self.res3 = self.__make_layer(128, 256, stride=2, reduction=reduction)
        self.res4 = self.__make_layer(256, 512, stride=2, reduction=reduction)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)
        
    def __make_layer(self, in_channels, out_channels, stride, reduction=8):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = self.downsample(in_channels, out_channels)
            
        return nn.Sequential(
            SEResidualBlock(in_channels, out_channels, reduction=reduction, downsample=downsample, stride=stride), 
            SEResidualBlock(out_channels, out_channels, reduction=reduction)
        )
        
    def forward(self, x):     
        x = self.layer(x)
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
        x = self.avgpool(x)
        x = rearrange(x, 'b c h w -> b (c h w)')
        x = self.linear(x)
        return x 
    
    def downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )