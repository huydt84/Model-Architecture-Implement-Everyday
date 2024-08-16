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