# Implement a model architecture everyday

## Motivation: 
When I took part in some competition in the past, I sometimes found it difficult to implement new and powerful model/block architectures (I even didn't know their existences). This repo is like a note for me - when I need to create new models, I can bring things from here instantly. Hope it helps you!

I'm trying to implement 1 new model/block/technique everyday (except weekends, housework is infinity). I'm busy doing other things everyday, so contribute to this repo consistently is so hard, but I will try =))

## Using the code
I haven't planned to make it a library yet - so you need to clone the repository to play with its stuff. Or just copy the code in some file and have fun!

## Usage
### Feature Pyramid Network (with Resnet18 bottom-up pathway) (11/8/2024)
```python
import torch
from models.fpn.fpn import FPN

channel = 3

m = FPN(image_channels=channel)
img = torch.rand(20, 3, 128, 256)   # (batch, channel, height, width)                                   
                                    # Make sure channel value is the same as the above variable.

outputs = m(img)
for output in outputs:
    print(output.shape)
```

### UNet (10/8/2024)
UNet is used for semantic segmentation, so I used `padding=1` in double convolution to keep the output shape unchanged

I modified the upsample (up-conv) process so that it has an additional parameter `mismatch_strategy`
- `mismatch_strategy = None`: before concatenation, the shape of 2 tensor stay the same. If **lenght** or **width** of input image is not divisible by 16, it will get error due to dimension mismatch.
- `mismatch_strategy = "crop"`: crop (the center of) the tensor with bigger size to be the same size as smaller one to prevent dimension mismatch. This is the method suggested in the original paper. 
- `mismatch_strategy = "pad" (recommended)`: pad the tensor with smaller size to be the same size as bigger one to prevent dimension mismatch. This will keep the feature map and the output shape unchanged. 

```python
import torch
from models.unet.unet import UNet

image_channels = 3
num_classes = 100

m = UNet(image_channels, num_classes, mismatch_strategy="pad")
img = torch.randn(20, 3, 158, 157)   # (batch, channel, height, width)                                   
                                     # Make sure channel value is the same as the above variable.

output = m(img)
print(b.size())   # torch.Size([20, 100, 158, 157])
```

### Squeeze-and-Excitation Networks (with ResNet18) (9/8/2024)
```python
import torch
from models.senet.se_block import  SEBlock1D, SEBlock2D
from models.senet.se_resnet import SEResNet18

channel = 3
num_classes = 1000

'''
SEBlock: Output shape is similar to input shape
'''
m = SEBlock1D(channel=512)
img = torch.rand(20, 512, 128)   # (batch, channel, length)                                   
                                 # Make sure channel value is the same as the above variable.

output = m(img)
print(output.size())   # torch.Size([20, 512, 128])

m = SEBlock2D(channel=1024, reduction=2)
img = torch.rand(20, 1024, 128, 256)   # (batch, channel, height, width)                                   
                                       # Make sure channel value is the same as the above variable.

output = m(img)
print(output.size())   # torch.Size([20, 1024, 128, 256])

'''
Add SEBlock in Residual Block of ResNet18
'''
m = SEResNet18(image_channels=channel, num_classes=num_classes, reduction=4)
img = torch.rand(20, 3, 128, 256)   # (batch, channel, height, width)                                   
                                    # Make sure channel value is the same as the above variable.

output = m(img)
print(output.size())   # torch.Size([batch, 1000])
```

### ResNet18 (8/8/2024)
```python
import torch
from models.resnet18.model import ResNet18

channel = 3
num_classes = 1000

m = ResNet18(image_channels=channel, num_classes=num_classes)
img = torch.rand(20, 3, 128, 256)   # (batch, channel, height, width)                                   
                                    # Make sure channel value is the same as the above variable.

output = m(img)
print(output.size())   # torch.Size([batch, 1000])
```

### VGG16 (7/8/2024)
```python
import torch
from models.vgg16.model import VGG16

channel = 3
num_classes = 1000

m = VGG16(image_channels=channel, num_classes=num_classes)
img = torch.rand(20, 3, 224, 224)   # (batch, channel, height, width)
                                    # in VGG, default input channel = 3, height = width = 224

output = m(img)
print(output.size())   # torch.Size([batch, 1000])
```

## Citation

```bibtex
@inproceedings{simonyan2015deep,
  title={Very Deep Convolutional Networks for Large-Scale Image Recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  booktitle={International Conference on Learning Representations},
  year={2015},
  url={https://arxiv.org/abs/1409.1556}
}
```

```bibtex
@inproceedings{7780459,
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Deep Residual Learning for Image Recognition}, 
  year={2016},
  pages={770-778}
}
```

```bibtex
@inproceedings{8578843,
  author={Hu, Jie and Shen, Li and Sun, Gang},
  booktitle={2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition}, 
  title={Squeeze-and-Excitation Networks}, 
  year={2018},
  pages={7132-7141},
}
```

```bibtex
@inproceedings{ronneberger2015u,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical image computing and computer-assisted intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```

```bibtex
@inproceedings{lin2017feature,
  title={Feature pyramid networks for object detection},
  author={Lin, Tsung-Yi and Dollár, Piotr and Girshick, Ross and He, Kaiming and Hariharan, Bharath and Belongie, Serge},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={2117--2125},
  year={2017}
}
```