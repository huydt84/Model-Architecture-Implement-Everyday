import torch
import torch.nn as nn

def initialize_weights(m):
    """Initialize the weights of convolutional, batch normalization, and linear layers"""

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
        
def shape_validation(x):
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    if len(x.shape) != 4:
        raise ValueError("Shape of input must be (batch, channel, height, width) or (channel, height, width). "
                            f"Your input shape currently is {x.shape}")
    return x 