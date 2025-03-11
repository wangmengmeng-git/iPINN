"""
iPINN_add1 for Diffusion-reaction equation  in 2D domain
"""

import scipy.io
import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from model import MLP
import random
from scipy import integrate

# Set default data type and random seed
dtype = torch.float32
torch.set_default_dtype(dtype)
seed_value = 33
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, only_inputs=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)


def iPINN_add1_Diff():
    configuration = {
        "Type": 'MLP',
        "Layers": 3,
        "Neurons": 20,
        "Activation": 'Tanh',
        "Optimizer": 'Adam',
        "Learning Rate": 1e-4,
        "Epochs": 100000,
        # delta t = delta x
        "x_number": 28,
        "t_number": 5,
        "wight": 10,
        "test": 1000
        # "LBFGS_Epochs_MAX_limit": 10000,
    }

    loss = torch.nn.MSELoss()
    u = MLP(2, 1, configuration['Layers'], configuration['Neurons'])

    # Optimizers
    opt = torch.optim.Adam(u.parameters(), lr=configuration['Learning Rate'])

    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.8)

    h = configuration["test"]
    xm = torch.linspace(-np.pi, np.pi, h)
    tm = torch.linspace(0, 1, h)
    xx, tt = torch.meshgrid(xm, tm)
    xc = xx.reshape(-1, 1).requires_grad_(True)
    tc = tt.reshape(-1, 1).requires_grad_(True)

    u_rea = torch.e ** (-tc) * (torch.sin(1 * xc) / 1 + torch.sin(2 * xc) / 2 + torch.sin(3 * xc) / 3 + torch.sin(
        4 * xc) / 4 + torch.sin(8 * xc) / 8)
    u_rea = u_rea.detach().numpy()
    ux_rea = torch.e ** (-tc) * (
            torch.cos(1 * xc) / 1 + torch.cos(2 * xc) + torch.cos(3 * xc) + torch.cos(4 * xc) + torch.cos(8 * xc))
    ux_rea = ux_rea.detach().numpy()
    ut_rea = -1 * torch.e ** (-tc) * (torch.sin(1 * xc) / 1 + torch.sin(2 * xc) / 2 + torch.sin(3 * xc) / 3 + torch.sin(
        4 * xc) / 4 + torch.sin(8 * xc) / 8)
    ut_rea = ut_rea.detach().numpy()


    # Output transformation function
    def output_transform(x, t, u):
        uxy = (x ** 2 - (torch.pi) ** 2) * (1 - (torch.e) ** (-t)) * u(torch.cat([x, t], dim=1)) + torch.sin(
            1 * x) / 1 + torch.sin(2 * x) / 2 + torch.sin(3 * x) / 3 + torch.sin(4 * x) / 4 + torch.sin(8 * x) / 8
        return uxy

    # Prediction function
    def predict_u(xc, tc, u):
        u_pred = output_transform(xc, tc, u)
        return u_pred.detach().numpy()

    def predict_u_x(xc,tc, u):
        return gradients(output_transform(xc,tc, u), xc, order=1).detach().numpy()

    def predict_u_t(xc,tc, u):
        return gradients(output_transform(xc,tc, u), tc, order=1).detach().numpy()

    # Training parameters
    epochs = configuration['Epochs']
    x_number = configuration['x_number']
    t_number = configuration['t_number']
    w = configuration['wight']
    resultlist = [0] * (x_number-1) * (t_number-1)


    def func(t, x):
        return np.exp(-t) * (
                3 / 2 * np.sin(2 * x) + 8 / 3 * np.sin(3 * x) + 15 / 4 * np.sin(4 * x) + 63 / 8 * np.sin(8 * x))
    for k in range(x_number-1):
        for v in range(t_number-1):
            x_lower, x_upper = -1 * np.pi + k * 2 * np.pi / (x_number-1), -1 * np.pi + 2 * (
                        k + 1) * np.pi / (x_number-1)

            t_lower, t_upper = 0. + v / (t_number-1), 0. + (v + 1) / (t_number-1)

            result, error = integrate.dblquad(func, x_lower, x_upper, lambda x: t_lower, lambda x: t_upper)
            resultlist[k*(t_number-1)+v] = result

    xx = torch.linspace(-1 * np.pi, np.pi, x_number)
    tt = torch.linspace(0, 1, t_number)

    X, T = torch.meshgrid(xx, tt)
    x = X.reshape(-1, 1).requires_grad_(True)
    t = T.reshape(-1, 1).requires_grad_(True)

    xx2_full = torch.linspace(-1 * np.pi, np.pi, x_number) + 0.5 * 2 * np.pi / (x_number - 1)
    xx2 = xx2_full[:-1]
    X2, T2 = torch.meshgrid(xx2, tt)
    x2 = X2.reshape(-1, 1).requires_grad_(True)
    t2 = T2.reshape(-1, 1).requires_grad_(True)

    #
    tt2_full = torch.linspace(0, 1, t_number) + 0.5 * 1 / (t_number - 1)
    tt2 = tt2_full[:-1]
    X3, T3 = torch.meshgrid(xx, tt2)
    x3 = X3.reshape(-1, 1).requires_grad_(True)
    t3 = T3.reshape(-1, 1).requires_grad_(True)

    k = torch.arange(0, x_number - 1).unsqueeze(1)  # 列向量
    j = torch.arange(0, t_number - 1)  # 行向量


    k1 = k * t_number + j
    k2 = (k + 1) * t_number + j
    k3 = (k + 1) * t_number + j + 1
    k4 = k * (t_number - 1)+j
    k5 = (k + 1) * (t_number - 1) +j
    k6 = k * t_number + j+1
    # Integration loss function
    def integrate_loss(x_number, t_number, k1, k2, k3, k4, k5, k6,uxy, uxy2,  u_x, u_x3):
        """
              Calculate the integration loss based on the Simpson's rule.
              """
        s1 = (1 / 6) * (2 * torch.pi / (x_number - 1)) * (
                1 * uxy[k2] + 4 * uxy2[k1] + 1 * uxy[k1])
        s2 = (1 / 6) * (2 * torch.pi / (x_number - 1)) * (
                1 * uxy[k3] + 4 * uxy2[k6] + 1 * uxy[k6])

        s3 = (1 / 6) * (1 / (t_number - 1)) * (
                1 * u_x[k6] + 4 * u_x3[k4] + 1 * u_x[k1])
        s4 = (1 / 6) * (1 / (t_number - 1)) * (
                1 * u_x[k3] + 4 * u_x3[k5] + 1 * u_x[k2])

        integrateloss = (s1 - s2 - s3 + s4).reshape(-1)
        result_tensor = torch.tensor(resultlist, dtype=integrateloss.dtype).reshape(integrateloss.shape)
        integrate_loss = (integrateloss + result_tensor) ** 2
        return integrate_loss

    # Training loop

    for i in range(epochs):
        opt.zero_grad()

        cond1 = torch.e ** (-t) * (
                3 / 2 * torch.sin(2 * x) + 8 / 3 * torch.sin(3 * x) + 15 / 4 * torch.sin(4 * x) + 63 / 8 * torch.sin(
            8 * x))
        uxy = output_transform(x, t, u)
        u_t = gradients(uxy, t, order=1)
        u_x = gradients(uxy, x, order=1)
        u_xx = gradients(u_x, x, order=1)

        uxy2 = output_transform(x2, t2, u)
        uxy3 = output_transform(x3, t3, u)
        u_x3 = gradients(uxy3, x3, order=1)

        # Calculate integration loss
        integ_loss = integrate_loss(x_number, t_number, k1, k2, k3, k4, k5, k6,uxy, uxy2,u_x, u_x3)

        #
        l1 = loss(u_t - u_xx, cond1)
        l = l1 + w * torch.sum(integ_loss)

        l.backward()
        opt.step()
        scheduler.step()

        if i % 1000 == 0:
            u_pred = predict_u(xc, tc, u)
            ux_pred = predict_u_x(xc, tc, u)
            ut_pred = predict_u_t(xc, tc, u)

            err = np.linalg.norm(u_rea - u_pred) / np.linalg.norm(u_rea)
            uxerr = np.linalg.norm(ux_rea - ux_pred, 2) / np.linalg.norm(ux_rea, 2)
            uterr = np.linalg.norm(ut_rea - ut_pred, 2) / np.linalg.norm(ut_rea, 2)

            print(f"Epoch: {i}, Loss: {l.item():.6e}")
            print(" uerror is: ", err)
            print(" uxerror is: ", uxerr)
            print(" uterror is: ", uterr)


    # Testing
    u_pred = predict_u(xc, tc, u)
    ux_pred = predict_u_x(xc, tc, u)
    ut_pred = predict_u_t(xc, tc, u)

    err = np.linalg.norm(u_rea - u_pred) / np.linalg.norm(u_rea)
    uxerr = np.linalg.norm(ux_rea - ux_pred, 2) / np.linalg.norm(ux_rea, 2)
    uterr = np.linalg.norm(ut_rea - ut_pred, 2) / np.linalg.norm(ut_rea, 2)

    print(f"Epoch: {i}, Loss: {l.item():.6e}")
    print(" uerror is: ", err)
    print(" uxerror is: ", uxerr)
    print(" uterror is: ", uterr)



if __name__ == "__main__":
    iPINN_add1_Diff()