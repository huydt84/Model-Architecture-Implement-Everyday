import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from typing import Literal

class SEBlock2D(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Hardsigmoid(inplace=True)   # replace sigmoid with h-sigmoid
        )

    def forward(self, x):
        y = x.mean(dim=(2, 3))   # (b c h w) -> (b c) 
        y = rearrange(self.layer(y), 'b c -> b c 1 1')
        return x * y
    
class ConvBnReLU(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1, groups=1, use_hs=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.Hardswish(inplace=True) if use_hs else nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)
    
class InvertedResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, expand_ratio, use_hs=True, use_se=True):
        super().__init__()
        self.stride = stride
        assert stride in [1,2]
        
        mid_channel = int(round(input_channel * expand_ratio))
        self.use_res_connect = self.stride==1 and input_channel==output_channel
        
        self.layer = nn.Sequential()
        if expand_ratio != 1:
            self.layer.add_module("conv1x1_bn_relu", ConvBnReLU(input_channel, mid_channel, kernel_size=1, use_hs=use_hs))
        self.layer.add_module("conv3x3_dw", ConvBnReLU(mid_channel, mid_channel, kernel_size, stride=stride, groups=mid_channel, use_hs=use_hs))
        if use_se:
            self.layer.add_module("se_block", SEBlock2D(mid_channel))
        self.layer.add_module("conv1x1", nn.Conv2d(mid_channel, output_channel, 1, 1, 0, bias=False))
        self.layer.add_module("bn", nn.BatchNorm2d(output_channel))
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.layer(x)
        else:
            return self.layer(x)
        
class MobileNetV3(nn.Module):
    def __init__(self, config, input_channel=3, num_classes=1000):
        super().__init__() 
        
        block_config = config.get("block_config", None)
        if not block_config:
            raise Exception("No block config found. Check the 'block_config' key in the config dictionary.")
        
        block_input_channel = config.get("input_channel", 16)
        block_last_channel = config.get("output_channel", 1024)
        
        layers = [ConvBnReLU(input_channel, block_input_channel, stride=2)]
        for k, t, c, use_se, use_hs, s in block_config:
            layers.append(InvertedResidualBlock(block_input_channel, c, kernel_size=k, 
                                                stride=s, expand_ratio=t, use_hs=use_hs, use_se=use_se))
            block_input_channel = c
                
        layers.append(ConvBnReLU(block_input_channel, block_input_channel*t, kernel_size=1)) 
        
        self.layer = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(block_input_channel*t, block_last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(block_last_channel, num_classes)
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
        x = self.classifier(x)
        return x

def mobilenet_v3(mode: Literal["large", "small"] = "small", input_channel=3, num_classes=1000):
    if mode == "large":
        config = {
            "block_config": [
                # k, t, c, SE, HS, s 
                [3,   1,  16, 0, 0, 1],
                [3,   4,  24, 0, 0, 2],
                [3,   3,  24, 0, 0, 1],
                [5,   3,  40, 1, 0, 2],
                [5,   3,  40, 1, 0, 1],
                [5,   3,  40, 1, 0, 1],
                [3,   6,  80, 0, 1, 2],
                [3, 2.5,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],
                [3, 2.3,  80, 0, 1, 1],
                [3,   6, 112, 1, 1, 1],
                [3,   6, 112, 1, 1, 1],
                [5,   6, 160, 1, 1, 2],
                [5,   6, 160, 1, 1, 1],
                [5,   6, 160, 1, 1, 1]
            ],
            "input_channel": 16,
            "output_channel": 1280
        }
    elif mode == "small":
        config = {
            "block_config": [
                # k, t, c, SE, HS, s 
                [3,    1,  16, 1, 0, 2],
                [3,  4.5,  24, 0, 0, 2],
                [3, 3.67,  24, 0, 0, 1],
                [5,    4,  40, 1, 1, 2],
                [5,    6,  40, 1, 1, 1],
                [5,    6,  40, 1, 1, 1],
                [5,    3,  48, 1, 1, 1],
                [5,    3,  48, 1, 1, 1],
                [5,    6,  96, 1, 1, 2],
                [5,    6,  96, 1, 1, 1],
                [5,    6,  96, 1, 1, 1],
            ],
            "input_channel": 16,
            "output_channel": 1024
        }
    else:
        raise ValueError(f"Unsupported model mode '{mode}'") 
    return MobileNetV3(config, input_channel, num_classes)
