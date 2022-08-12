import argparse
import jsonschema
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import yaml

config_schema = {
    "type": "object",
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "name": {"type":"string"},
                "annotations": {
                    "type": "object",
                    "additionalProperties": {"type":"string"}
                }
            }
        },
        "data": {
            "type": "object",
            "properties": {
                "src": {"type": "string"}
            }
        },
        "layers": {
            "type": "array",
            "item": {
                "type": "object",
                "properties": {
                    "type": {"type":"string"},
                    "size": {"type":"integer"}
                }
            }
        },
        "training": {
            "device": {"type": "string"},
            "epochs": {"type": "integer"},
            "optimizer": {"type": "string"}
        }
    }
}

def get_data(src: str, train: bool) -> tuple:
    if src == 'Fashion-MNIST':
        return datasets.FashionMNIST(root='data', train=train, download=True, transform=ToTensor())

def get_config(filename: str = 'config.yaml') -> dict:
    with open(filename) as f:
        config = yaml.safe_load(f)
        jsonschema.validate(instance=config, schema=config_schema)
        return config

def train(args):
    config = get_config()
    print(config)

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

