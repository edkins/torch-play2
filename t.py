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

config_schema = {
    "type": "object",
    "properties": {
        "version": {"const":"0.1"},
        "metadata": {
            "type": "object",
            "properties": {
                "name": {"type":"string"},
                "annotations": {
                    "type": "object",
                    "additionalProperties": {"type":"string"}
                }
            },
            "required": ["name"]
        },
        "data": {
            "type": "object",
            "properties": {
                "src": {"type": "string"}
            },
            "required": ["src"]
        },
        "layers": {
            "type": "array",
            "item": {
                "type": "object",
                "properties": {
                    "type": {"type":"string"},
                    "size": {
                        "type":["integer","string"],
                        "regex": "^[0-9]+(x[0-9]+)*$"
                    }
                },
                "required": ["type"]
            }
        },
        "training": {
            "properties": {
                "batch_size": {"type": "integer"},
                "device": {"type": "string"},
                "epochs": {"type": "integer"},
                "optimizer": {
                    "type": "object",
                    "properties": {
                        "type": {"type":"string"}
                    }
                },
                "loss": {
                    "type": "object",
                    "properties": {
                        "type": {"type":"string"}
                    }
                }
            },
            "required": ["epochs","batch_size"]
        }
    },
    "required": ["metadata","data","layers","training","version"]
}

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

def get_config(filename: str = 'config.yaml') -> tuple[str,dict]:
    with open(filename) as f:
        text = f.read()
        config = yaml.safe_load(text)
        jsonschema.validate(instance=config, schema=config_schema)
        return text, config

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
            elif typ == 'relu':
                submodules.append(nn.ReLU())
                if shape != None:
                    if shape != x_shape:
                        raise Exception("relu can't change shape")
            elif typ == 'softmax':
                submodules.append(nn.Softmax(dim=1))
                if shape != None:
                    if shape != x_shape:
                        raise Exception("softmax can't change shape")
            else:
                raise Exception(f"Unrecognized layer type: {typ}")
        self.stack = nn.Sequential(*submodules)

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

def serve(args):
    pass

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

