import torch
import torch.nn as nn

class ConvBnReLU(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU6(inplace=True)   # original paper used ReLU6
        )
    def forward(self, x):
        return self.layer(x)
    
class InvertedResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1,2]
        
        mid_channel = int(round(input_channel * expand_ratio))
        self.use_res_connect = self.stride==1 and input_channel==output_channel
        
        self.layer = nn.Sequential()
        if expand_ratio != 1:
            self.layer.add_module("conv1x1_bn_relu", ConvBnReLU(input_channel, mid_channel, kernel_size=1))
        self.layer.add_module("conv3x3_dw", ConvBnReLU(mid_channel, mid_channel, stride=stride, groups=mid_channel))
        self.layer.add_module("conv1x1", nn.Conv2d(mid_channel, output_channel, 1, 1, 0, bias=False))
        self.layer.add_module("bn", nn.BatchNorm2d(output_channel))
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.layer(x)
        else:
            return self.layer(x)
        
class MobileNetV2(nn.Module):
    def __init__(self, input_channel=3, num_classes=1000, block_config=None):
        super().__init__()
        
        if not block_config:
            block_config = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ] 
        
        block_input_channel = 32
        block_last_channel = 1280
        
        layers = [ConvBnReLU(input_channel, block_input_channel, stride=2)]
        for t, c, n, s in block_config:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidualBlock(block_input_channel, c, stride, expand_ratio=t))
                block_input_channel = c
                
        layers.append(ConvBnReLU(block_input_channel, block_last_channel, kernel_size=1)) 
        
        self.layer = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear( block_last_channel, num_classes)
        )
        
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
        x = self.layer(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
        
        
        