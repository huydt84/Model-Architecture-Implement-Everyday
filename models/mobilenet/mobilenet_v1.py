import torch
import torch.nn as nn

class ConvBnReLU(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

class ConvBnReLU_DW(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=stride, padding=1, groups=input_channel, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)    
    

class MobileNetV1(nn.Module):
    def __init__(self, input_channel, num_classes):
        super().__init__()
        self.conv_layer = nn.Sequential(
            ConvBnReLU(input_channel, 32, 2),
            ConvBnReLU_DW(32, 64, 1),
            ConvBnReLU_DW(64, 128, 2),
            ConvBnReLU_DW(128, 128, 1),
            ConvBnReLU_DW(128, 256, 2),
            ConvBnReLU_DW(256, 256, 1),
            ConvBnReLU_DW(256, 512, 2),
            ConvBnReLU_DW(512, 512, 1),
            ConvBnReLU_DW(512, 512, 1),
            ConvBnReLU_DW(512, 512, 1),
            ConvBnReLU_DW(512, 512, 1),
            ConvBnReLU_DW(512, 512, 1),
            ConvBnReLU_DW(512, 1024, 2),
            ConvBnReLU_DW(1024, 1024, 1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x