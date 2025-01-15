import torch
import numpy as np
from typing import Optional, Tuple, List, Union

from backend import NDArray, Device, fn
from ..autograd import Tensor
from .functional import *

from copy import deepcopy

__all__ = [
    'Module',
    'Sigmoid',
    'ReLU',
    'Linear',
    'Conv2d',
    'MaxPool2d',
    'CELoss'
]

class Module():
    """Base abstract class for all NN modules."""
    def forward():
        raise NotImplementedError()
    
    # Modules do not need backward(), as it is handled by autograd.

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def parameters(self) -> List[Tensor]:
        return [p.copy() for p in self.__dict__.values() if isinstance(p, Tensor) and p.requires_grad]

    def param_num(self) -> int:
        return sum([p.size for p in self.parameters()])
    
    def init(self):
        raise NotImplementedError() #TODO: Implement initialization methods.
    
    def update(self):
        for p in self.__dict__.values():
            if isinstance(p, Tensor) and p.requires_grad:
                p -= 0.1 * p.grad
    

"""Element-wise activation layers"""
class Sigmoid(Module):
    """Sigmoid activation layer."""
    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)
        

class ReLU(Module):
    """ReLU activation layer."""
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class Linear(Module):
    """Fully-connected layers."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = 1 / in_features ** 0.5   # Kaiming initialization
        self.weight = Tensor.rand((in_features, out_features), -bound, bound, requires_grad=True)
        self.bias = Tensor.rand((out_features,), -bound, bound, requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight, self.bias)


class Conv2d(Module):
    """2D convolutional layers with kernel size 3x3, stride 1, and padding 1."""
    #TODO: Update the conv2d CUDA implementation to support arbitrary k, s and p.
    def __init__(self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        assert kernel_size == 3 and stride == 1 and padding == 1, "Currently only support kernel_size=3, stride=1, padding=1"    
    
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel = Tensor.rand((out_channels, in_channels, kernel_size, kernel_size), 
                                  requires_grad=True)
        self.bias = Tensor.rand((out_channels,), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.kernel, self.bias)

class MaxPool2d(Module):
    """2D pooling layers with kernel size 2x2, stride 2."""
    #TODO: Update the pooling2d CUDA implementation to support arbitrary k, s.
    def __init__(self,
                 kernel_size: int = 2,
                 stride: int = 2
    ):
        assert kernel_size == 2 and stride == 2, "Currently only support kernel_size=2, stride=2"
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return maxpool2d(x, self.kernel_size, self.stride)


"""Softmax and loss functions"""
class CELoss(Module):
    """Softmax and Cross-Entropy Loss."""
    def forward(self, 
                logits: Tensor, # (N, C)
                labels: Tensor  # (N,)
    ) -> Tensor:
        return celoss(logits, labels)
    
    
    
