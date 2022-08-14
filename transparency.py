from datetime import datetime
import multiprocessing
import numpy as np
import time
import torch
import yaml

from configured_nn import ConfiguredNN
from model import get_config, get_results, get_state_dict

def create_highly_activating_input(model_name: str, layers: list[dict], state_dict: dict, x_shape: tuple[int], layer: str, neuron: str, device: str) -> np.ndarray:
    def f(t):
        return t.tanh() * 0.5 + 0.5
        #return t.clamp(0,1)
    print(f'Creating highly activating input for model {model_name}, layer {layer}, neuron {neuron}')
    start_time = time.monotonic()
    model = ConfiguredNN(layers, x_shape).to(device)
    model.load_state_dict(state_dict)
    inp = torch.rand(1, *x_shape, requires_grad=True, device=device)
    optimizer = torch.optim.SGD([inp], lr=0.01)
    model.eval()
    model.requires_grad_(False)
    output_index = (0, *parse_neuron(neuron))
    maximum = 50000
    date = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    for i in range(maximum+1):
        #model.zero_grad()
        output = model.get_neuron_output(f(inp), int(layer), output_index)
        if int(layer) == len(layers) - 1:
            value = -output.log()
        else:
            value = -output
        optimizer.zero_grad()
        value.backward()
        optimizer.step()
        if i % 1000 == 0 or i == maximum:
            print(value.item())
            inp_np = f(inp)[0].detach().to('cpu').numpy()
            time_taken = time.monotonic() - start_time
            test_name = f'test-{layer}-{neuron}-{date}'
            test_result = {
                'name': test_name,
                'x_shape': x_shape,
                'data': inp_np.tolist(),
                'model': model_name,
                'layer': layer,
                'neuron': neuron,
                'time_taken': time_taken,
                'steps': i,
                'finished': i == maximum
            }
            filename = f'models/{model_name}/{test_name}.yaml'
            with open(filename, 'w') as file:
                file.write(yaml.dump(test_result))
    print(f'Created highly activating input for model {model_name}, layer {layer}, neuron {neuron}')

def get_test(filename: str):
    with open(filename) as f:
        text = f.read()
        return yaml.safe_load(text)

def parse_neuron(neuron: str) -> tuple[int]:
    return tuple(int(x) for x in neuron.split(','))

def create_and_run_test(model: str, layer: str, neuron: str) -> str:
    _, config = get_config(f'models/{model}/config.yaml')
    results = get_results(f'models/{model}/results.yaml')
    state_dict = get_state_dict(f'models/{model}/params.pickle')
    process = multiprocessing.Process(target=create_highly_activating_input, kwargs={
        'model_name': model,
        'layers':config['layers'], 
        'state_dict': state_dict,
        'x_shape': results['x_shape'],
        'layer': layer,
        'neuron': neuron,
        'device': config['training']['device']
    })
    process.start()
