import argparse
from datetime import datetime
import jsonschema
import multiprocessing
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
from configured_nn import ConfiguredNN

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

def train(args):
    config_text, config = get_config()
    data_src = config['data']['src']
    batch_size = config['training']['batch_size']
    device = config['training'].get('device','cpu')
    num_epochs = config['training']['epochs']
    name = config['metadata']['name']

    train_data = get_data(data_src, train=True)
    test_data = get_data(data_src, train=False)
    classes = list(train_data.classes)

    x_shape = tuple(train_data[0][0].shape)
    y_shape = (len(classes),)
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
    results['classes'] = classes
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
    multiprocessing.set_start_method('spawn')
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

