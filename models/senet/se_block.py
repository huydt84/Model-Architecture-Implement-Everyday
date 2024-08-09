import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class SEBlock2D(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = x.mean(dim=(2, 3))   # (b c h w) -> (b c) 
        y = rearrange(self.layer(y), 'b c -> b c 1 1')
        return x * y
    
class SEBlock1D(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = x.mean(dim=2)   # (b c l) -> (b c) 
        y = self.layer(y).unsqueeze(2)
        return x * y
