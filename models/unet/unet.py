import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)
    
class DownScale(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2D(in_channels, out_channels, mid_channels)
        )

    def forward(self, x):
        return self.layer(x)
    
class UpScale(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, mismatch_strategy:Literal[None, "pad", "crop"]=None):
        super().__init__()
        self.mismatch_strategy = mismatch_strategy

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv2D(in_channels, out_channels, mid_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_height = x2.size()[2] - x1.size()[2]
        diff_width = x2.size()[3] - x1.size()[3]

        if self.mismatch_strategy == "pad":
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
            # See the "Padding size" part for more understanding
            x1 = F.pad(x1, [diff_width // 2, diff_width - diff_width // 2,
                            diff_height // 2, diff_height - diff_height // 2])
        if self.mismatch_strategy == "crop":
            x2 = torch.narrow(x2, 2, diff_height // 2, x1.size()[2])
            x2 = torch.narrow(x2, 3, diff_width // 2, x1.size()[3])

        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, image_channels=3, num_classes=1000, mid_channels=None, mismatch_strategy:Literal[None, "pad", "crop"]="pad"):
        super(UNet, self).__init__()
        self.image_channels = image_channels
        self.num_classes = num_classes
        self.mismatch_strategy = mismatch_strategy

        self.conv = DoubleConv2D(image_channels, 64, mid_channels)
        self.down1 = DownScale(64, 128, mid_channels)
        self.down2 = DownScale(128, 256, mid_channels)
        self.down3 = DownScale(256, 512, mid_channels)
        self.down4 = DownScale(512, 1024, mid_channels)
        self.up1 = UpScale(1024, 512, mid_channels, mismatch_strategy)
        self.up2 = UpScale(512, 256, mid_channels, mismatch_strategy)
        self.up3 = UpScale(256, 128, mid_channels, mismatch_strategy)
        self.up4 = UpScale(128, 64, mid_channels, mismatch_strategy)
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
        
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
            
        if (x.size()[2] % 16 != 0 or x.size()[3] % 16 != 0) and self.mismatch_strategy is None:
            raise ValueError("Input height or width is not divisible by 16. \
                             Change input shape or change mismatch_strategy to 'pad' or 'crop'.")
        x1 = self.conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out(x)