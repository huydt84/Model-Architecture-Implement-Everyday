# Implement a model architecture everyday

## Motivation: 
When I took part in some competition in the past, I sometimes found it difficult to implement new and powerful model/block architectures (I even didn't know their existences). This repo is like a note for me - when I need to create new models, I can bring things from here instantly. Hope it helps you!

I'm trying to implement 1 new model/block/technique everyday. I'm busy doing other things everyday, so contribute to this repo consistently is so hard, but I will try =))

## Using the code
I haven't planned to make it a library yet - so you need to clone the repository to play with its stuff.

## Usage
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