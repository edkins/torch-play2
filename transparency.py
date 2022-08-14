import numpy as np
import torch

import configured_nn

def create_highly_activating_input(layers: list[dict], x_shape: tuple[int], layer: int, neuron: int) -> np.ndarray:
    model = ConfiguredNN(layers[:layer+1], x_shape)
    inp = torch.rand(x_shape, requires_grad=True)
    opt = torch.optim.Adam([inp])
    for i in range(10):
        model.zero_grad()
        optimizer.zero_grad()
        output_layer = model(inp)
        output_layer[neuron].backward()
        optimizer.step()
    return inp.detach().to('cpu').numpy()
