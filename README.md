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
| alpha = 0.25  | DL | DL | DL | DL |
| alpha = 0.50  | DL | DL | DL | DL |
| alpha = 0.75  | DL | DL | DL | DL |
| alpha = 1.00  | DL | DL | DL | DL |

## With top (1000 classes, imagenet)

|  | 128px | 160px | 192px | 224px |
| ------------- | :---: | :---: | :---: | :---: |
| alpha = 0.25  | DL | DL | DL | DL |
| alpha = 0.50  | DL | DL | DL | DL |
| alpha = 0.75  | DL | DL | DL | DL |
| alpha = 1.00  | DL | DL | DL | DL |


