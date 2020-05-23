# MobileNet-v1-Pytorch
MobileNet v1 implementation in Pytorch. Pretrained weights converted from Keras implementation.

# Usage

```
from mobilenet_v1 import MobileNet_v1
model = MobileNet_v1(1000, alpha=0.25, input_size=128, include_top=False)
```

# Pretrained weights

You can obtain pretrained imagenet weights using code in [convert_weights_keras_2_torch.py](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/blob/master/convert_weights_keras_2_torch.py) or download from table below.

## No top

|  | 128px | 160px | 192px | 224px |
| ------------- | :---: | :---: | :---: | :---: |
| alpha = 0.25  | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_128_alpha_0.25_no_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_160_alpha_0.25_no_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_192_alpha_0.25_no_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_224_alpha_0.25_no_top.pth) |
| alpha = 0.50  | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_128_alpha_0.5_no_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_160_alpha_0.5_no_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_192_alpha_0.5_no_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_224_alpha_0.5_no_top.pth) |
| alpha = 0.75  | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_128_alpha_0.75_no_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_160_alpha_0.75_no_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_192_alpha_0.75_no_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_224_alpha_0.75_no_top.pth) |
| alpha = 1.00  | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_128_alpha_1.0_no_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_160_alpha_1.0_no_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_192_alpha_1.0_no_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_224_alpha_1.0_no_top.pth) |

## With top (1000 classes, imagenet)

|  | 128px | 160px | 192px | 224px |
| ------------- | :---: | :---: | :---: | :---: |
| alpha = 0.25  | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_128_alpha_0.25_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_160_alpha_0.25_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_192_alpha_0.25_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_224_alpha_0.25_top.pth) |
| alpha = 0.50  | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_128_alpha_0.5_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_160_alpha_0.5_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_192_alpha_0.5_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_224_alpha_0.5_top.pth) |
| alpha = 0.75  | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_128_alpha_0.75_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_160_alpha_0.75_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_192_alpha_0.75_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_224_alpha_0.75_top.pth) |
| alpha = 1.00  | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_128_alpha_1.0_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_160_alpha_1.0_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_192_alpha_1.0_top.pth) | [DL](https://github.com/ZFTurbo/MobileNet-v1-Pytorch/releases/download/v1.0/mobilenet_v1_size_224_alpha_1.0_top.pth) |



