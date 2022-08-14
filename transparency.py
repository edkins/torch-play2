from datetime import datetime
import numpy as np
import time
import torch
import yaml

from configured_nn import ConfiguredNN
from model import get_config, get_results

def create_highly_activating_input(layers: list[dict], x_shape: tuple[int], layer: int, neuron: tuple[int], device: str) -> np.ndarray:
    model = ConfiguredNN(layers[:layer+1], x_shape).to(device)
    inp = torch.rand(*x_shape, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([inp])
    model.eval()
    for i in range(10):
        model.zero_grad()
        optimizer.zero_grad()
        output_layer = model(inp)
        (-output_layer[neuron]).backward()
        optimizer.step()
    return inp.detach().to('cpu').numpy()

def get_test(filename: str):
    with open(filename) as f:
        text = f.read()
        return yaml.safe_load(text)

def parse_neuron(neuron: str) -> tuple[int]:
    return tuple(int(x) for x in neuron.split(','))

def create_and_run_test(model: str, layer: str, neuron: str) -> str:
    _, config = get_config(f'models/{model}/config.yaml')
    results = get_results(f'models/{model}/results.yaml')
    print(f'Creating highly activating input for model {model}, layer {layer}, neuron {neuron}')
    start_time = time.monotonic()
    inp = create_highly_activating_input(config['layers'], results['x_shape'], int(layer), parse_neuron(neuron), config['training']['device'])
    time_taken = time.monotonic() - start_time
    print(f'Created highly activating input for model {model}, layer {layer}, neuron {neuron}')
    date = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    test_name = f'test-{layer}-{neuron}-{date}'
    test_result = {
        'name': test_name,
        'x_shape': results['x_shape'],
        'data': inp.tolist(),
        'model': model,
        'layer': layer,
        'neuron': neuron,
        'time_taken': time_taken
    }
    filename = f'models/{model}/{test_name}.yaml'
    with open(filename, 'w') as f:
        f.write(yaml.dump(test_result))
