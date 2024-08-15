import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = self.layer(x)
        x = self.downsample(x)
        out = F.relu(out + x)
        return x
    
class FPN(nn.Module):
    def __init__(self, image_channels=3, mid_channels=None):
        super().__init__()
        self.mid_channels = mid_channels
        
        # Normal sequential convolution layer
        self.layer = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Bottom-up layers
        self.botlayer1 = self.__make_layer(64, 256, stride=1)
        self.botlayer2 = self.__make_layer(256, 512, stride=2)
        self.botlayer3 = self.__make_layer(512, 1024, stride=2)
        self.botlayer4 = self.__make_layer(1024, 2048, stride=2)
        
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        
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
        
    def __make_layer(self, in_channels, out_channels, stride):   
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=stride, mid_channels=self.mid_channels), 
            ResidualBlock(out_channels, out_channels, mid_channels=self.mid_channels)
        )
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        return F.interpolate(x, size=(y.size()[2], y.size()[3]), mode='bilinear') + y
        
    def forward(self, x):  
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if len(x.shape) != 4:
            raise ValueError("Shape of input must be (batch, channel, height, width) or (channel, height, width). "
                             f"Your input shape currently is {x.shape}")
               
        x = self.layer(x)
        
        bot1 = self.botlayer1(x)
        bot2 = self.botlayer2(bot1)
        bot3 = self.botlayer3(bot2)
        bot4 = self.botlayer4(bot3)
        
        # Top-down
        p4 = self.toplayer(bot4)
        p3 = self._upsample_add(p4, self.latlayer1(bot3))
        p2 = self._upsample_add(p3, self.latlayer2(bot2))
        p1 = self._upsample_add(p2, self.latlayer3(bot1))
        
        # Smooth
        p1 = self.smooth1(p1)
        p2 = self.smooth2(p2)
        p3 = self.smooth3(p3)
        return p1, p2, p3, p4