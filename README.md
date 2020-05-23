# MobileNet-v1-Pytorch
MobileNet v1 implementation in Pytorch. Pretrained weights converted from Keras implementation.

# Usage

```
from mobilenet_v1 import MobileNet_v1
model = MobileNet_v1(1000, alpha=0.25, input_size=128, include_top=False)
```

# Pretrained weights

You can obtain weights using code in convert_weights_keras_2_torch.py or download from table below.