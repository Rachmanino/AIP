'''
    Optimizers
'''

import torch
import numpy as np
from typing import Optional, Tuple, List, Union
from copy import deepcopy

from backend import NDArray, Device, fn
from ..autograd import Tensor
from .modules import Module

__all__ = ['Optimizer', 'SGD', 'Adam']

class Optimizer:
    def __init__(self, module: Module, lr: float):
        self.module = module
        self.lr = lr

    def step(self):
        for param in self.module.parameters():
            self._update(param)

    def zero_grad(self):
        for param in self.module.parameters():
            param.zero_grad()

    def _update(self, param: Tensor):
        raise NotImplementedError
    

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer
    """
    
    def __init__(self, module: Module, lr: float):
        super().__init__(module, lr)

    def _update(self, param: Tensor):
        param = (param - self.lr * param.grad).detach() # detach the tensor from the computation graph


class Adam(Optimizer):
    """
    Adam optimizer
    """
    def __init__(self, module: Module, lr: float, beta1: float=0.9, beta2: float=0.999, eps: float=1e-8):
        super().__init__(module, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [Tensor(np.zeros_like(param.data)) for param in self.params]
        self.v = [Tensor(np.zeros_like(param.data)) for param in self.params]

    def _update(self, param: Tensor):
        self.t += 1
        idx = 0
        for param, m, v in zip(self.params, self.m, self.v):
            m = self.beta1 * m + (1 - self.beta1) * param.grad
            v = self.beta2 * v + (1 - self.beta2) * param.grad**2
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            param = (param - self.lr * m_hat / (v_hat ** 0.5 + self.eps)).detach()
            idx += 1
    
    def reset(self):
        self.t = 0
        self.m = [Tensor(np.zeros_like(param.data)) for param in self.params]
        self.v = [Tensor(np.zeros_like(param.data)) for param in self.params]