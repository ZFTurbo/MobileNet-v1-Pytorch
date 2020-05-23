# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo), IPPM RAS'


import os
import torch
import numpy as np
from keras.applications import MobileNet
from mobilenet_v1 import MobileNet_v1


def convertMobileNetWeights(out_path, input_size=224, alpha=0.25, include_top=True):
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    model_k = MobileNet(input_shape=(input_size, input_size, 3), alpha=alpha, weights='imagenet', include_top=include_top, pooling='avg')
    print(model_k.summary())
    model_t = MobileNet_v1(1000, alpha=alpha, input_size=input_size, include_top=include_top)

    res_t = dict()
    st = model_t.state_dict()
    for i, el in enumerate(st):
        arr = el.split('.')
        print(arr)
        if 'model' in el:
            key = (int(arr[1]), int(arr[2]))
            if key not in res_t:
                res_t[key] = []
            res_t[key].append((el, st[el].numpy().shape))
        elif 'fc.':
            key = (100, )
            if key not in res_t:
                res_t[key] = []
            res_t[key].append((el, st[el].numpy().shape))

    res_torch = dict()
    for i, el in enumerate(sorted(list(res_t.keys()))):
        print(i, el, res_t[el])
        res_torch[i] = res_t[el]
        print(res_torch[i])

    total = 0
    res_k = dict()
    for level_id in range(len(model_k.layers)):
        layer = model_k.layers[level_id]
        layer_type = layer.__class__.__name__
        if layer_type in ['Conv2D', 'BatchNormalization', 'DepthwiseConv2D', 'Dense']:
            w = layer.get_weights()
            print('{} {} {} {}'.format(total, level_id, layer_type, w[0].shape))
            res_k[total] = [level_id, layer_type, w[0].shape]

            # Modify state_dict
            if layer_type == 'Conv2D':
                weigths_t = w[0].transpose((3, 2, 1, 0))
                torch_name = res_torch[total][0][0]
                torch_shape = res_torch[total][0][1]
                print('Modify: {}'.format(torch_name))
                # Check shape
                if weigths_t.shape != torch_shape:
                    print('Shape mismatch: {} != {}'.format(weigths_t.shape, torch_shape))
                st[torch_name] = torch.from_numpy(weigths_t)
                if len(res_torch[total]) == 2:
                    print('Store bias...')
                    weigths_t = w[1]
                    torch_name = res_torch[total][1][0]
                    torch_shape = res_torch[total][1][1]
                    print('Modify: {}'.format(torch_name))
                    # Check shape
                    if weigths_t.shape != torch_shape:
                        print('Shape mismatch: {} != {}'.format(weigths_t.shape, torch_shape))
                    st[torch_name] = torch.from_numpy(weigths_t)

            elif layer_type == 'DepthwiseConv2D':
                weigths_t = w[0].transpose((2, 3, 1, 0))
                torch_name = res_torch[total][0][0]
                torch_shape = res_torch[total][0][1]
                print('Modify: {}'.format(torch_name))
                # Check shape
                if weigths_t.shape != torch_shape:
                    print('Shape mismatch: {} != {}'.format(weigths_t.shape, torch_shape))
                st[torch_name] = torch.from_numpy(weigths_t)
            elif layer_type == 'BatchNormalization':
                for i in range(4):
                    weigths_t = w[i]
                    torch_name = res_torch[total][i][0]
                    torch_shape = res_torch[total][i][1]
                    print('Modify: {}'.format(torch_name))
                    # Check shape
                    if weigths_t.shape != torch_shape:
                        print('Shape mismatch: {} != {}'.format(weigths_t.shape, torch_shape))
                    st[torch_name] = torch.from_numpy(weigths_t)

            total += 1

    model_t.load_state_dict(st)

    data_k = np.random.uniform(-1, 1, (100, input_size, input_size, 3)).astype(np.float32)
    data_t = data_k.transpose((0, 3, 2, 1))
    print(data_k.shape, data_t.shape)

    pred_k = model_k.predict(data_k)
    data_t = torch.from_numpy(data_t)
    model_t.eval()
    with torch.no_grad():
        pred_t = model_t(data_t)
    if include_top:
        pred_t = pred_t.numpy()
    else:
        pred_t = pred_t.permute(0, 3, 2, 1).squeeze().numpy()

    print(pred_k.shape, pred_t.shape)
    diff = (pred_t - pred_k)
    print(diff.min(), diff.max(), diff.mean())

    if np.abs(diff).max() > 0.001:
        print('Large error!')
        exit()

    top = '_no_top'
    if include_top:
        top = '_top'
    torch.save(model_t.state_dict(), out_path + "mobilenet_v1_size_{}_alpha_{}{}.pth".format(input_size, alpha, top))


if __name__ == '__main__':
    out_path = 'mobilenet_v1_torch/'
    for input_size in [128, 160, 192, 224]:
        for alpha in [0.25, 0.50, 0.75, 1.0]:
            for include_top in [False, True]:
                convertMobileNetWeights(out_path, input_size, alpha, include_top)