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
    def __init__(self, input_channel=3, num_classes=1000):
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
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if len(x.shape) != 4:
            raise ValueError("Shape of input must be (batch, channel, height, width) or (channel, height, width). "
                             f"Your input shape currently is {x.shape}")
        x = self.conv_layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x