import argparse
from datetime import datetime
import jsonschema
import numpy as np
import os
import pickle
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Optional, Union
import yaml

from serve import serve
from model import get_config

def get_data(src: str, train: bool) -> tuple:
    if src == 'Fashion-MNIST':
        return datasets.FashionMNIST(root='data', train=train, download=True, transform=ToTensor())
    else:
        raise Exception(f"Dataset not known: {src}")

def get_optimizer(parameters, opt: Optional[dict]):
    if opt == None or 'type' not in opt:
        return torch.optim.Adam(parameters)
    elif opt['type'] == 'adam':
        return torch.optim.Adam(parameters)
    else:
        raise Exception(f"Optimizer not known: {opt['type']}")

def get_loss(loss: Optional[dict]):
    if loss == None or 'type' not in loss:
        return torch.nn.CrossEntropyLoss()
    elif loss['type'] == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()
    else:
        raise Exception(f"Loss function not known: {loss['type']}")

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

def train(args):
    config_text, config = get_config()
    data_src = config['data']['src']
    batch_size = config['training']['batch_size']
    device = config['training'].get('device','cpu')
    num_epochs = config['training']['epochs']
    name = config['metadata']['name']

    train_data = get_data(data_src, train=True)
    test_data = get_data(data_src, train=False)

    x_shape = tuple(train_data[0][0].shape)
    y_shape = (len(train_data.classes),)
    model = ConfiguredNN(config['layers'], x_shape).to(device)
    optimizer = get_optimizer(model.parameters(), config['training'].get('optimizer'))
    loss_fn = get_loss(config['training'].get('loss'))
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True, pin_memory_device=device, persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8, pin_memory=True, pin_memory_device=device, persistent_workers=True)

    results = {'epochs':[]}
    initial_start_time = time.monotonic()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        start_time = time.monotonic()
        training_loop(train_dataloader, model, loss_fn, optimizer, device)
        accuracy = test_loop(test_dataloader, model, loss_fn, device)
        time_taken = time.monotonic() - start_time
        print(f"Time taken for epoch: {time_taken}")
        results['epochs'].append({'time_taken': time_taken, 'accuracy': accuracy})
    results['accuracy'] = accuracy
    results['time_taken'] = time.monotonic() - initial_start_time
    results['layers'] = model.layer_results
    results['x_shape'] = list(x_shape)
    results['y_shape'] = list(y_shape)
    save_model(name, config_text, model, results)

def save_model(name: str, config_text: str, model, results):
    date = datetime.now().strftime('%Y-%m-%dT%H-%M')
    path = f'models/{name}--{date}'
    os.makedirs(path)
    with open(f'{path}/config.yaml', 'w') as f:
        f.write(config_text)
    with open(f'{path}/params.pickle', 'wb') as f:
        pickle.dump(model.state_dict(), f)
    with open(f'{path}/results.yaml', 'w') as f:
        f.write(yaml.dump(results))
    print(f"Saved model to {path}/")

def training_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print(f"Batch {batch}/{num_batches}")

def test_loop(dataloader, model, loss_fn, device) -> float:
    size = len(dataloader.dataset)
    #num_batches = len(dataloader)
    model.eval()
    correct = torch.tensor(0, dtype=torch.float).to(device)
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum()
    correct /= size
    print(f"Accuracy: {100*correct.item()}%")
    return correct.item()

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(func=train)
    parser_serve = subparsers.add_parser('serve')
    parser_serve.set_defaults(func=serve)
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()

