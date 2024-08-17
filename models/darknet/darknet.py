import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import initialize_weights, shape_validation

class ConvBnReLU(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)

def conv3_max(input_channel):
    return nn.Sequential(
        ConvBnReLU(input_channel, input_channel*2, kernel_size=3),
        ConvBnReLU(input_channel*2, input_channel, kernel_size=1),
        ConvBnReLU(input_channel, input_channel*2, kernel_size=3),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    
def conv5(input_channel):
    return nn.Sequential(
        ConvBnReLU(input_channel, input_channel*2, kernel_size=3),
        ConvBnReLU(input_channel*2, input_channel, kernel_size=1),
        ConvBnReLU(input_channel, input_channel*2, kernel_size=3),
        ConvBnReLU(input_channel*2, input_channel, kernel_size=1),
        ConvBnReLU(input_channel, input_channel*2, kernel_size=3)
    )
    
class ResidualBlock(nn.Module):
    def __init__(self,input_channels):
        super().__init__()
        self.conv1 = ConvBnReLU(input_channels, input_channels // 2, kernel_size=1)
        self.conv2 = ConvBnReLU(input_channels // 2, input_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return out
    
class DarkNet19(nn.Module):
    def __init__(self, input_channels, num_classes, init_weight=True):
        super().__init__()
        self.layer = nn.Sequential(
            ConvBnReLU(input_channels, 32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            ConvBnReLU(32, 64, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            conv3_max(64),
            conv3_max(128),
            
            conv5(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv5(512)
        )
        
        self.classifier = nn.Sequential(
            ConvBnReLU(1024, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        if init_weight:
            self.apply(initialize_weights)
            
    def forward(self, x):
        x = shape_validation(x)
        x = self.layer(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        x = F.softmax(x, 1)
        
        return x
    
class DarkNet53(nn.Module):
    def __init__(self, input_channels, num_classes, init_weight=True):
        super().__init__()
        self.layer = nn.Sequential(
            ConvBnReLU(input_channels, 32, kernel_size=3),
            ConvBnReLU(32, 64, kernel_size=3, stride=2),    
            self.__make_layer(in_channels=64, num_blocks=1),
            
            ConvBnReLU(64, 128, kernel_size=3, stride=2),    
            self.__make_layer(in_channels=128, num_blocks=2),
            
            ConvBnReLU(128, 256, kernel_size=3, stride=2),    
            self.__make_layer(in_channels=256, num_blocks=8),
            
            ConvBnReLU(256, 512, kernel_size=3, stride=2),    
            self.__make_layer(in_channels=512, num_blocks=8),
            
            ConvBnReLU(512, 1024, kernel_size=3, stride=2),    
            self.__make_layer(in_channels=1024, num_blocks=4),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(1024, num_classes)
        
        if init_weight:
            self.apply(initialize_weights)
        
    def forward(self, x):
        x = shape_validation(x)
        x = self.layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = F.softmax(x, 1)
        
        return x
        
    
    def __make_layer(self, in_channels, num_blocks):
        list_block = [ResidualBlock(in_channels)] * num_blocks   
        return nn.Sequential(*list_block)