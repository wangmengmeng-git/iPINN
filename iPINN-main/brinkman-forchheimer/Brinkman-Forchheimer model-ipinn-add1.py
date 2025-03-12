"""
iPINN_add1 for  Brinkman-Forchheimer model in 1D domain
"""
import scipy.io
import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from model2 import MLP
import random

# Set default data type and random seed
dtype = torch.float32
torch.set_default_dtype(dtype)
seed_value = 1
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, only_inputs=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

def iPINN_add1_bf():
    configuration = {"Type": 'MLP',
                     "Layers": 3,
                     "Neurons": 20,
                     "Activation": 'Tanh',
                     "Optimizer": 'Adam',
                     "Learning Rate": 1e-4,
                     "Epochs": 50000,
                     "N_domain": 35,
                     "Domain": "x -> [0, 1]",
                     'data num':5,
                     "wight": 1,
                     "h": 1000
                     }


    loss = torch.nn.MSELoss()
    u = MLP(1, 1, configuration['Layers'], configuration['Neurons'])
    opt = torch.optim.Adam(u.parameters(), lr=configuration['Learning Rate'])

    v_e=u.v_e
    epochs = configuration['Epochs']
    N = configuration['N_domain']
    num=configuration['data num']
    w=configuration['wight']

    def output_transform(x,  u):
        uxy = (x * (1 - x) * u(x))
        return uxy

    def predict_u(xc, u):
        u_pred = output_transform(xc, u)
        return u_pred.detach().numpy()

    def predict_u_x(xc, u):
        return gradients(output_transform(xc, u), xc, order=1).detach().numpy()

    def gen_traindata(num):
        ob_x  = torch.linspace(1 / (num + 1), num / (num + 1), num)
        return ob_x

    h = configuration["h"]
    xc = torch.linspace(0, 1, h)[:, np.newaxis].requires_grad_(True)
    r = (v * e / (1e-3 * K)) ** (0.5)
    u_rea = g * K / v * (1 - torch.cosh(r * (xc - H / 2)) / np.cosh(r * H / 2))
    u_real = u_rea.detach().numpy()
    ux_rea = -g * K / v * ((torch.sinh(r * (xc - H / 2)) * (r)) / np.cosh(r * H / 2))
    ux_real = ux_rea.detach().numpy()

    ob_x = gen_traindata(num)[:, np.newaxis].requires_grad_(True)
    x = torch.linspace(0, 1, N)[:, np.newaxis].requires_grad_(True)
    cond1 = torch.ones_like(x)

    for i in range(epochs):

        opt.zero_grad()
        x2 = (x[:-1] + x[1:]) / 2
        ob_u = g * K / v * (1 - torch.cosh(r * (ob_x - H / 2)) / np.cosh(r * H / 2))

        uxy =output_transform(x,  u)
        uxy2 = output_transform(x2,  u)
        u_x = gradients(uxy, x, order=1)
        u_xx = gradients(u_x, x, order=1)

        """
                   Calculate the integration loss based on the Simpson's rule.
                   """
        llist = ((v_e / e) * (u_x[:-1] - u_x[1:]) + (1 / 6) * (v / K) * (uxy[:-1] + 4 * uxy2 + uxy[1:]) * (
                    1 / (N - 1)) - 1 / (N - 1)) ** 2
        l1 = loss((-v_e / e) * u_xx + v * uxy / K, cond1)
        l = l1 +  w*(torch.sum(llist)) +  loss(output_transform(ob_x,u), ob_u)

        l.backward()
        opt.step()

        if i%1000==0:
            u_pred = predict_u(xc, u)
            ux_pred = predict_u_x(xc, u)
            err = np.linalg.norm(u_real - u_pred, 2) / np.linalg.norm(u_real, 2)
            uxerr = np.linalg.norm(ux_real - ux_pred, 2) / np.linalg.norm(ux_real, 2)
            print(i)
            print('ve Error', abs((v_e.item() - 1e-3) / 1e-3))
            print('u  Error: %.3e' % ( err))
            print('u_x  Error: %.3e' % (uxerr))

    u_pred = predict_u(xc, u)
    ux_pred = predict_u_x(xc, u)
    err = np.linalg.norm(u_real - u_pred, 2) / np.linalg.norm(u_real, 2)
    uxerr = np.linalg.norm(ux_real - ux_pred, 2) / np.linalg.norm(ux_real, 2)
    print('ve Error', abs((v_e.item() - 1e-3) / 1e-3))
    print('u  Error: %.3e' % (err))
    print('u_x  Error: %.3e' % (uxerr))

if __name__ == "__main__":
    g = 1
    v = 1e-3
    K = 1e-3
    e = 0.4
    H = 1
    iPINN_add1_bf()

