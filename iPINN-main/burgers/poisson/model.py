import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_neurons, activation=torch.tanh):
        super(MLP, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.act_func = activation
        self.layers = nn.ModuleList()
        self.layer_input = nn.Linear(self.in_features, self.num_neurons)
        self.layer_input.weight.data.uniform_(-math.sqrt(6 / (self.in_features + self.num_neurons)),
                                              math.sqrt(6 / (self.in_features + self.num_neurons)))

        for ii in range(self.num_layers - 1):
            layer = nn.Linear(self.num_neurons, self.num_neurons)

            layer.weight.data.uniform_(-math.sqrt(6 / (self.num_neurons + self.num_neurons)),
                                       math.sqrt(6 / (self.num_neurons + self.num_neurons)))
            self.layers.append(layer)

        self.layer_output = nn.Linear(self.num_neurons, self.out_features)

        self.layer_output.weight.data.uniform_(-math.sqrt(6 / (self.num_neurons + self.out_features)),
                                               math.sqrt(6 / (self.num_neurons + self.out_features)))

    def forward(self, x):
        x = self.act_func(self.layer_input(x))
        for dense in self.layers:
            x = self.act_func(dense(x))
        x = self.layer_output(x)
        return x

