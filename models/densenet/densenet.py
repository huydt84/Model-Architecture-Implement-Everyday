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
    
class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.add_module('dense_layer_%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
    
class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        return self.down_sample(x)

