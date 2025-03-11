
"""
PINN for burgers equation in 2D domain
"""

import scipy.io
import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from model import MLP
import random

# Set default data type and random seed
dtype = torch.float32
torch.set_default_dtype(dtype)
seed_value = 34
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, only_inputs=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

# Main training function
def PINN_bur():
    configuration = {
        "Type": 'MLP',
        "Layers": 4,
        "Neurons": 32,
        "Activation": 'Tanh',
        "Learning Rate": 1e-3,
        "Epochs": 20000,
        # delta t =4* delta x
        "x_number": 18 * 8,
        "t_number": 18,
        "wight": 100,
        "LBFGS_Epochs_MAX_limit": 10000,
    }

    loss = torch.nn.MSELoss()
    u = MLP(2, 1, configuration['Layers'], configuration['Neurons'])

    # Optimizers
    opt = torch.optim.Adam(u.parameters(), lr=configuration['Learning Rate'])
    lbfgs_optimizer = torch.optim.LBFGS(u.parameters(), lr=1, max_iter=configuration["LBFGS_Epochs_MAX_limit"], max_eval=None,
                                        tolerance_grad=1e-20, tolerance_change=1e-20,
                                        history_size=100, line_search_fn='strong_wolfe')

    # Load data
    data = scipy.io.loadmat('burgers.mat')
    tc_np = data['t'].flatten()[:, None]
    xc_np = data['x'].flatten()[:, None]
    Exact = data['usol']
    Exact_u = np.real(Exact)

    #
    X, T = np.meshgrid(xc_np, tc_np)
    xc, tc = (torch.from_numpy(X.flatten()[:, None]).float(), torch.from_numpy(T.flatten()[:, None]).float())
    u_star = Exact_u.T.flatten()[:, None]

    # Output transformation function
    def output_transform(x, t, u):
        uxy = (1 - x) * (1 + x) * (1 - torch.exp(-t)) * u(torch.cat([x, t], dim=1)) - torch.sin(np.pi * x)
        return uxy

    # Prediction function
    def predict(xc, tc, u):
        u_pred = output_transform(xc, tc, u)
        return u_pred.detach().numpy()

    # Training parameters
    epochs = configuration['Epochs']
    x_number = configuration['x_number']
    t_number = configuration['t_number']
    w = configuration['wight']

    #
    xx = torch.linspace(-1, 1, x_number)
    tt = torch.linspace(0, 1, t_number)
    X, T = torch.meshgrid(xx, tt)
    x = X.reshape(-1, 1).requires_grad_(True)
    t = T.reshape(-1, 1).requires_grad_(True)


    # Training loop

    for i in range(epochs):
        opt.zero_grad()


        uxy = output_transform(x, t, u)
        u_t = gradients(uxy, t, order=1)
        u_x = gradients(uxy, x, order=1)
        u_xx = gradients(u_x, x, order=1)


        l1 = loss(u_t + uxy * u_x - 0.01 / np.pi * u_xx, torch.zeros_like(uxy))
        l = l1


        l.backward()
        opt.step()

        if i % 100 == 0:
            u_pred = predict(xc, tc, u)
            err = np.linalg.norm(u_star - u_pred) / np.linalg.norm(u_star)
            print(f"Epoch: {i}, Loss: {l.item():.6e}, Error: {err:.6e}")

    # LBFGS optimizer closure function
    def closure():
        lbfgs_optimizer.zero_grad()

        uxy = output_transform(x, t, u)
        u_t = gradients(uxy, t, order=1)
        u_x = gradients(uxy, x, order=1)
        u_xx = gradients(u_x, x, order=1)


        l1 = loss(u_t + uxy * u_x - 0.01 / np.pi * u_xx, torch.zeros_like(uxy))
        l = l1

        l.backward()
        return l

    # LBFGS optimization
    print("Starting LBFGS optimization...")
    lbfgs_optimizer.step(closure)

    # Testing
    u_pred = predict(xc, tc, u)
    err = np.linalg.norm(u_star - u_pred) / np.linalg.norm(u_star)
    print("u_error is: ", err)


if __name__ == "__main__":
    PINN_bur()