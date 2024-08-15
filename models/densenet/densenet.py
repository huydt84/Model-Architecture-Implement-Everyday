import torch
import torch.nn as nn
import torch.nn.functional as F

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
        concated_feature = torch.cat(x, 1)
        new_feature = self.layer(concated_feature)
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

class DenseNet(nn.Module):
    
    # Densenet-BC model (default is densenet121)
    def __init__(self, image_channels=3, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, reduction=2, drop_rate=0, num_classes=100):

        super().__init__()

        # First convolution
        self.features = nn.Sequential(
            nn.Conv2d(image_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Each dense block
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('dense_block_%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_input_features=num_features,
                                    num_output_features=num_features // reduction)
                self.features.add_module('transition_%d' % (i + 1), trans)
                num_features = num_features // reduction

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

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
            
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
def densenet121(image_channels=3):
    return DenseNet(image_channels, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64)

def densenet161(image_channels=3):
    return DenseNet(image_channels, growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96)

def densenet169(image_channels=3):
    return DenseNet(image_channels, growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64)

def densenet201(image_channels=3):
    return DenseNet(image_channels, growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64)