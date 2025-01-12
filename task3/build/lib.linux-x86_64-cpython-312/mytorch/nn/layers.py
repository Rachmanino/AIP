import torch
import numpy as np
from typing import Optional, Tuple, List, Union

from backend import NDArray, Device, fn
from ..tensor import Tensor

from copy import deepcopy

__all__ = [
    'Module',
    'Linear',
    'Conv2d',
    'MaxPool2d',
    'Sigmoid',
    'ReLU',
    'CELoss'
]

class Module():
    """Base abstract class for all NN modules."""
    def forward():
        raise NotImplementedError()
    
    def backward():
        raise NotImplementedError()
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

#TODO: Replace all '=' in following backward functions with '+=' after implementing autodiff

"""Common NN-layers"""
class Linear(Module):
    """Fully-connected layers."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor.rand((in_features, out_features), requires_grad=True)
        self.bias = Tensor.rand((out_features,), requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        self.batchsize = x.shape[0]
        self.output = Tensor((self.batchsize, self.out_features), x.device)
        fn.fc_fwd(x, self.weight, self.bias, self.output)
        return self.output
    
    def backward(self, out_grad: Tensor) -> Tensor:
        assert out_grad.shape == (self.batchsize, self.out_features), "Output gradient shape mismatch!"
        fn.fc_bwd(self.input, self.weight, self.bias, out_grad, self.input.grad, self.weight.grad, self.bias.grad)


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
        assert x.shape[1] == self.in_channels, "Input channel mismatch!"
        self.input = x
        self.output = Tensor((x.shape[0], self.out_channels, x.shape[2], x.shape[3]), x.device)
        fn.conv2d_k33p1s1_fwd(x, self.kernel, self.bias, self.output)
        return self.output

    def backward(self, out_grad: Tensor) -> Tensor:
        assert out_grad.shape == self.output.shape, "Output gradient shape mismatch!"
        fn.conv2d_k33p1s1_bwd(self.input, self.kernel, out_grad, self.input.grad, self.kernel.grad, self.bias.grad)

class MaxPool2d(Module):
    """2D pooling layers with kernel size 2x2, stride 2."""
    #TODO: Update the pooling2d CUDA implementation to support arbitrary k, s.
    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        self.output = Tensor((x.shape[0], x.shape[1], x.shape[2]//2, x.shape[3]//2), x.device)
        fn.maxpooling2d_k22s2_fwd(x, self.output)
        return self.output

    def backward(self, out_grad: Tensor) -> Tensor:
        assert out_grad.shape == self.output.shape, "Output gradient shape mismatch!"
        fn.maxpooling2d_k22s2_bwd(self.input, self.output, out_grad, self.input.grad)



"""Element-wise activation layers"""
class Sigmoid(Module):
    """Sigmoid activation function."""
    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        self.output = Tensor(x.shape, x.device)
        fn.sigmoid_fwd(x, self.output)
        return self.output
    
    def backward(self, out_grad: Tensor) -> Tensor:
        assert out_grad.shape == self.output.shape, "Output gradient shape mismatch!"
        fn.sigmoid_bwd(self.input, out_grad, self.input.grad) 
        
class ReLU(Module):
    """ReLU activation function."""
    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        self.output = Tensor(x.shape, x.device)
        fn.relu_fwd(x, self.output)
        return self.output
    
    def backward(self, out_grad: Tensor):
        assert out_grad.shape == self.output.shape, "Output gradient shape mismatch!"
        fn.relu_bwd(self.input, out_grad, self.input.grad) 



"""Softmax and loss functions"""
class CELoss(Module):
    """Softmax and Cross-Entropy Loss."""
    def forward(self, 
                logits: Tensor, # (N, C)
                labels: Tensor  # (N,)
    ) -> Tensor:
        self.input = logits
        self.labels = labels
        self.probs = Tensor(logits.shape, logits.device)
        fn.softmax_fwd(logits, self.probs)
        self.output = fn.celoss_fwd(self.probs, labels)
        return self.output

    def backward(self, out_grad = 1): # out_grad is always 1
        fn.softmax_ce_bwd(self.probs, self.labels, out_grad, self.input.grad)
    
    
    
