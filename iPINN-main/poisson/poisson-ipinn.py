"""
iPINN for poisson in 1D domain
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
from model import MLP
import random


# Set default data type and random seed
dtype = torch.float32
torch.set_default_dtype(dtype)
seed_value = 94
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

def lr_lambda(epoch):
    if epoch <= 5000:
        return 2
    else:
        return 2 * 0.9 ** ((epoch - 5000) // 400)

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

def iPINN_po():
    configuration = {
        "Type": 'MLP',
        "Layers": 3,
        "Neurons": 20,
        "Activation": 'Tanh',
        "Optimizer": 'Adam',
        "Learning Rate": 1e-3,
        "Epochs": 20000,
        "N_domain": 20,
        "wight": 5,
        "test": 1000
    }

    # Output transformation function
    def output_transform(xc, u):
        return xc * (torch.pi - xc) * u(xc) + xc

    # Prediction functions
    def predict_u(xc, u):
        u_pred = output_transform(xc, u)
        return u_pred.detach().numpy()

    def predict_u_x(xc, u):
        return gradients(output_transform(xc, u), xc, order=1).detach().numpy()

    loss = torch.nn.MSELoss()
    u = MLP(1, 1, configuration['Layers'], configuration['Neurons'])

    # Optimizers
    opt = torch.optim.Adam(u.parameters(), lr=configuration['Learning Rate'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    h = configuration["test"]
    xc = torch.linspace(0, np.pi, h)[:, None].requires_grad_(True)

    # Real solution and its derivative
    u_rea = xc + torch.sin(1 * xc) / 1 + torch.sin(2 * xc) / 2 + torch.sin(3 * xc) / 3 + torch.sin(4 * xc) / 4 + torch.sin(8 * xc) / 8
    u_real = u_rea.detach().numpy()
    ux_rea = 1 + torch.cos(1 * xc) / 1 + torch.cos(2 * xc) + torch.cos(3 * xc) + torch.cos(4 * xc) + torch.cos(8 * xc)
    ux_real = ux_rea.detach().numpy()

    w = configuration['wight']
    epochs = configuration['Epochs']
    N = configuration['N_domain']
    x = torch.linspace(0, np.pi, N)[:, None].requires_grad_(True)

    #
    condlist = []
    for j in range(N - 1):
        cos_terms_j = (
            np.cos((np.pi / (N - 1)) * j) +
            np.cos(2 * (np.pi / (N - 1)) * j) +
            np.cos(3 * (np.pi / (N - 1)) * j) +
            np.cos(4 * (np.pi / (N - 1)) * j) +
            np.cos(8 * (np.pi / (N - 1)) * j)
        )
        cos_terms_j1 = (
            np.cos((np.pi / (N - 1)) * (j + 1)) +
            np.cos(2 * (np.pi / (N - 1)) * (j + 1)) +
            np.cos(3 * (np.pi / (N - 1)) * (j + 1)) +
            np.cos(4 * (np.pi / (N - 1)) * (j + 1)) +
            np.cos(8 * (np.pi / (N - 1)) * (j + 1))
        )
        cond11 = -(cos_terms_j1) + cos_terms_j
        condlist.append(cond11)
    condlisttensor = torch.tensor(condlist).view(-1, 1)

    # Training loop
    for i in range(epochs):
        opt.zero_grad()
        uxy = output_transform(x, u)
        ux = gradients(uxy, x, order=1)
        u_x = gradients(ux, x, order=1)
        cond1 = - (8 * torch.sin(8 * x) + 1 * torch.sin(1 * x) + 2 * torch.sin(2 * x) + 3 * torch.sin(3 * x) + 4 * torch.sin(4 * x))
        llist = (ux[:-1] - ux[1:] - condlisttensor) ** 2
        l1 = loss(u_x, cond1)
        l = l1 + w * torch.sum(llist)
        l.backward()
        opt.step()
        scheduler.step()

        if i % 100 == 0:
            u_pred = predict_u(xc, u)
            ux_pred = predict_u_x(xc, u)
            err = np.linalg.norm(u_real - u_pred, 2) / np.linalg.norm(u_real, 2)
            uxerr = np.linalg.norm(ux_real - ux_pred, 2) / np.linalg.norm(ux_real, 2)
            print(f'Epoch: {i}, L2 Error: {err:.3e}, uxerror: {uxerr:.3e}')

    # Final evaluation
    u_pred = predict_u(xc, u)
    ux_pred = predict_u_x(xc, u)
    err = np.linalg.norm(u_real - u_pred, 2) / np.linalg.norm(u_real, 2)
    uxerr = np.linalg.norm(ux_real - ux_pred, 2) / np.linalg.norm(ux_real, 2)
    print(f'Final L2 Error: {err:.3e}')
    print(f'Final uxerror: {uxerr:.3e}')

if __name__ == "__main__":
    iPINN_po()