import numpy as np
from torch import nn
from typing import Optional, Union

def parse_shape(size: Union[int,str,None]) -> Optional[tuple[int]]:
    if size == None:
        return None
    elif isinstance(size, int):
        return (size,)
    else:
        return tuple(int(x) for x in size.split('x'))

def shape_product(shape: tuple[int]) -> int:
    return int(np.product(shape))


class ConfiguredNN(nn.Module):
    def __init__(self, layers: list[dict], x_shape: tuple[int]):
        super().__init__()
        submodules = []
        layer_results = []
        for layer in layers:
            typ = layer['type']
            shape = parse_shape(layer.get('size'))
            if typ == 'dense':
                if len(x_shape) > 1:
                    # input to nn.Linear must be flat
                    submodules.append(nn.Flatten())
                    x_shape = (shape_product(x_shape),)
                if shape == None:
                    raise Exception("Must specify size for dense layer")
                submodules.append(nn.Linear(x_shape[0], shape_product(shape)))
                if len(shape) > 1:
                    submodules.append(nn.Unflatten(1, shape))
                x_shape = shape
            elif typ == 'conv':
                if len(x_shape) != 3:
                    raise Exception(f"conv can currently only be used on 3d tensors. Got {x_shape}")
                stride = layer.get('stride',1)
                kernel_size = layer['kernel_size']
                padding = 0
                dilation = 1
                d, h, w = x_shape
                h = (h + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
                w = (w + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
                if shape == None:
                    raise Exception("Size must be specified for conv")
                if shape[1] != h and shape[2] != w:
                    raise Exception(f"Expected {h}x{w}. Got {shape[1]}x{shape[2]}")
                x_shape = (shape[0], h, w)
                submodules.append(nn.Conv2d(
                    in_channels=d,
                    out_channels=shape[0],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation))
            elif typ == 'relu':
                submodules.append(nn.ReLU())
                if shape != None and shape != x_shape:
                    raise Exception("relu can't change shape")
            elif typ == 'softmax':
                submodules.append(nn.Softmax(dim=1))
                if shape != None and shape != x_shape:
                    raise Exception("softmax can't change shape")
            else:
                raise Exception(f"Unrecognized layer type: {typ}")
            layer_results.append({'type':typ, 'shape':list(x_shape)})
        self.stack = nn.Sequential(*submodules)
        self.layer_results = layer_results

    def forward(self, x):
        return self.stack(x)

