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
def iPINN_add1_bur():
    configuration = {
        "Type": 'MLP',
        "Layers": 4,
        "Neurons": 32,
        "Activation": 'Tanh',
        "Optimizer": 'Adam',
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

    #
    xx2 = (torch.linspace(-1, 1, x_number) + 0.5 * 2 / (x_number - 1))[:-1]
    X2, T2 = torch.meshgrid(xx2, tt)
    x2 = X2.reshape(-1, 1).requires_grad_(True)
    t2 = T2.reshape(-1, 1).requires_grad_(True)

    #
    tt2 = (torch.linspace(0, 1, t_number) + 0.5 * 1 / (t_number - 1))[:-1]
    X3, T3 = torch.meshgrid(xx, tt2)
    x3 = X3.reshape(-1, 1).requires_grad_(True)
    t3 = T3.reshape(-1, 1).requires_grad_(True)

    # Indices
    chuyi = 1
    k_indices = torch.arange(0, x_number - 1, chuyi).unsqueeze(1)
    j_indices = torch.arange(0, t_number - 1, chuyi)

    k1 = k_indices * t_number + j_indices
    k2 = (k_indices + chuyi) * t_number + j_indices
    k3 = (k_indices + chuyi) * t_number + j_indices + chuyi
    k4 = k_indices * (t_number - 1)
    k5 = (k_indices + chuyi) * (t_number - 1)

    # Integration loss function
    def integrate_loss(x_number, t_number, k1, k2, k3, k4, k5, uxy, uxy2, uxy3, u_x, u_x3):

        s1 = (2 / (x_number - 1)) * (
                (1 / 6) * uxy[k2] +
                (2 / 3) * uxy2[k1] +
                (1 / 6) * uxy[k1]
        )

        s2 = (2 / (x_number - 1)) * (
                (1 / 6) * uxy[k3] +
                (2 / 3) * uxy2[k1 + chuyi] +
                (1 / 6) * uxy[k1 + chuyi]
        )

        s5 = 0.5 * (1 / (t_number - 1)) * (
                (1 / 6) * uxy[k1 + chuyi] ** 2 +
                (2 / 3) * uxy3[k4 + j_indices] ** 2 +
                (1 / 6) * uxy[k1] ** 2
        )

        s6 = 0.5 * (1 / (t_number - 1)) * (
                (1 / 6) * uxy[k3] ** 2 +
                (2 / 3) * uxy3[k5 + j_indices] ** 2 +
                (1 / 6) * uxy[k2] ** 2
        )

        s3 = (0.01 / np.pi) * (1 / (t_number - 1)) * (
                (1 / 6) * u_x[k1 + chuyi] +
                (2 / 3) * u_x3[k4 + j_indices] +
                (1 / 6) * u_x[k1]
        )

        s4 = (0.01 / np.pi) * (1 / (t_number - 1)) * (
                (1 / 6) * u_x[k3] +
                (2 / 3) * u_x3[k5 + j_indices] +
                (1 / 6) * u_x[k2]
        )


        integration_loss = ((-s1 + s2 - s5 + s6 + s3 - s4)) ** 2
        return integration_loss

    # Training loop


    for i in range(epochs):
        opt.zero_grad()

        uxy = output_transform(x, t, u)
        u_t = gradients(uxy, t, order=1)
        u_x = gradients(uxy, x, order=1)
        u_xx = gradients(u_x, x, order=1)

        uxy2 = output_transform(x2, t2, u)
        uxy3 = output_transform(x3, t3, u)
        u_x3 = gradients(uxy3, x3, order=1)

        # Calculate integration loss
        integ_loss = integrate_loss(x_number, t_number, k1, k2, k3, k4, k5, uxy, uxy2, uxy3, u_x, u_x3)

        #
        l1 = loss(u_t + uxy * u_x - 0.01 / np.pi * u_xx, torch.zeros_like(uxy))
        l = l1 + w * torch.sum(integ_loss)


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

        uxy2 = output_transform(x2, t2, u)
        uxy3 = output_transform(x3, t3, u)
        u_x3 = gradients(uxy3, x3, order=1)

        integ_loss = integrate_loss(x_number, t_number, k1, k2, k3, k4, k5, uxy, uxy2, uxy3, u_x, u_x3)
        l1 = loss(u_t + uxy * u_x - 0.01 / np.pi * u_xx, torch.zeros_like(uxy))
        l = l1 + w * torch.sum(integ_loss)

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
    iPINN_add1_bur()