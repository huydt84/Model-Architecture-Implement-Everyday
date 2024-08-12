import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.layer = nn.Sequential()
        self.layer.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.layer.add_module('relu1', nn.ReLU(inplace=True)),
        self.layer.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.layer.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.layer.add_module('relu2', nn.ReLU(inplace=True)),
        self.layer.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        if drop_rate > 0:
            self.layer.add_module('dropout', nn.Dropout(p=drop_rate, inplace=True))
        
    def forward(self, x):
        new_feature = self.layer(x)
        return new_feature
