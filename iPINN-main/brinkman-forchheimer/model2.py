import torch
import torch.nn as nn
import math


class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_neurons, activation=torch.tanh):
        super(MLP, self).__init__()

        self.v_e = nn.Parameter(torch.nn.functional.softplus(torch.tensor(0.0, requires_grad=True)) * 0.1)
        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.act_func = activation

        self.layers = nn.ModuleList()
        self.layer_input = nn.Linear(self.in_features, self.num_neurons)

        nn.init.xavier_normal_(self.layer_input.weight)

        for ii in range(self.num_layers - 1):
            layer = nn.Linear(self.num_neurons, self.num_neurons)

            nn.init.xavier_normal_(layer.weight)
            self.layers.append(layer)

        self.layer_output = nn.Linear(self.num_neurons, self.out_features)

        nn.init.xavier_normal_(self.layer_output.weight)

    def forward(self, x):
        x = self.act_func(self.layer_input(x))
        for dense in self.layers:
            x = self.act_func(dense(x))
        x = self.layer_output(x)
        return x


