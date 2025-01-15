"""
    Functions w/ autograd support
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Union

from backend import NDArray, Device, fn
from ..autograd import Tensor, Op

__all__ = [
    'relu',
    'sigmoid',
    'linear',
    'conv2d',
    'maxpool2d',
    'celoss',
]

# Classes starting with _ are internal classes and should not be used directly.

class _ReLU(Op):
    def compute(self, x: NDArray) -> NDArray:
        output = NDArray(list(x.shape), x._device)
        fn.relu_fwd(x, output)
        return output
    
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        in_grad = Tensor(grad.shape, grad.device)
        fn.relu_bwd(node.inputs[0], grad, in_grad)
        return in_grad
def relu(x: Tensor) -> Tensor:
    return _ReLU()(x)


class _Sigmoid(Op):
    def compute(self, x: NDArray) -> NDArray:
        output = NDArray(list(x.shape), x._device)
        fn.sigmoid_fwd(x, output)
        return output
    
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        in_grad = Tensor(grad.shape, grad.device)
        fn.sigmoid_bwd(node.inputs[0], grad, in_grad)
        return in_grad
    
def sigmoid(x: Tensor) -> Tensor:
    return _Sigmoid()(x)

class _Linear(Op):
    def compute(self, x: NDArray, weight: NDArray, bias: NDArray) -> NDArray:
        assert x.shape[1] == weight.shape[0]
        assert weight.shape[1] == bias.shape[0]
        output = NDArray([x.shape[0], weight.shape[1]], x._device)
        fn.fc_fwd(x, weight, bias, output)
        return output
    
    def gradient(self, grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x, weight, bias = node.inputs
        in_grad = Tensor(grad.shape, grad.device)
        weight_grad = Tensor(weight.shape, weight.device)
        bias_grad = Tensor(bias.shape, bias.device)
        fn.fc_bwd(x, weight, bias, grad, in_grad, weight_grad, bias_grad)
        return in_grad, weight_grad, bias_grad
    
def linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    return _Linear()(x, weight, bias)

class _Conv2d(Op):
    def compute(self, x: NDArray, kernel: NDArray, bias: NDArray) -> NDArray:
        assert x.shape[1] == kernel.shape[1], "Input channel mismatch!"
        assert kernel.shape[0] == bias.shape[0], "Bias channel mismatch!"
        output = NDArray([x.shape[0], kernel.shape[0], x.shape[2], x.shape[3]], x._device)
        fn.conv2d_k33p1s1_fwd(x, kernel, bias, output)
        return output
    
    def gradient(self, grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x, kernel, bias = node.inputs
        in_grad = Tensor(x.shape, x.device)
        kernel_grad = Tensor(kernel.shape, kernel.device)
        bias_grad = Tensor(bias.shape, bias.device)
        fn.conv2d_k33p1s1_bwd(x, kernel, grad, in_grad, kernel_grad, bias_grad)
        return in_grad, kernel_grad, bias_grad
def conv2d(x: Tensor, kernel: Tensor, bias: Tensor) -> Tensor:
    return _Conv2d()(x, kernel, bias)

class _MaxPool2d(Op):
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        assert kernel_size == 2 and stride == 2, "Currently only support kernel_size=2, stride=2"
        self.kernel_size = kernel_size
        self.stride = stride

    def compute(self, x: NDArray) -> NDArray:
        self.output = NDArray([x.shape[0], x.shape[1], x.shape[2]//2, x.shape[3]//2], x._device)
        fn.maxpool2d_k22s2_fwd(x, self.output)
        return self.output
    
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        x = node.inputs[0]
        in_grad = Tensor(x.shape, x.device)
        fn.maxpool2d_k22s2_bwd(x, self.output, grad, in_grad)
        return in_grad
def maxpool2d(x: Tensor, kernel_size: int = 2, stride: int = 2) -> Tensor:
    return _MaxPool2d(kernel_size, stride)(x)

class _CELoss(Op):
    def compute(self, logits: NDArray, labels: NDArray) -> NDArray:
        self.probs = NDArray(list(logits.shape), logits._device)
        fn.softmax_fwd(logits, self.probs)
        return fn.celoss_fwd(self.probs, labels)
    
    def gradient(self, grad: Tensor, node: Tensor) -> Tuple[Tensor, Tensor]:
        assert isinstance(grad, Tensor) # out_grad should be provided as a tensor!
        in_grad = Tensor(node.inputs[0].shape, node.inputs[0].device)
        fn.softmax_ce_bwd(self.probs, node.inputs[1], grad, in_grad)
        return in_grad, None # No gradient w.r.t. labels
def celoss(logits: Tensor, labels: Tensor) -> Tensor:
    return _CELoss()(logits, labels)




