import pytest
from tensor import Tensor
from nn import Sigmoid, ReLU

import torch
from torch.nn import functional as F

tol = 1e-5

def test_sigmoid():
    f = Sigmoid()
    x = Tensor.rand((2,3,4), requires_grad=True)
    y = f(x)
    assert y.shape == x.shape
    assert y.device == x.device
    x_ref = x.torch()
    y_ref = F.sigmoid(x_ref)
    assert y_ref.requires_grad
    torch.testing.assert_close(y.torch(), y_ref, rtol=tol, atol=tol)

    y_ref.backward(torch.ones_like(y_ref))
    f.backward(Tensor.ones(y.shape))
    torch.testing.assert_close(x.grad.torch(), x_ref.grad, rtol=tol, atol=tol)

def test_relu():
    f = ReLU()
    x = Tensor.rand((2,3,4), -1, 1, requires_grad=True) # U(-1, 1)
    y = f(x)
    assert y.shape == x.shape
    assert y.device == x.device
    x_ref = x.torch()
    y_ref = F.relu(x_ref)
    assert y_ref.requires_grad
    torch.testing.assert_close(y.torch(), y_ref, rtol=tol, atol=tol)

    y_ref.backward(torch.ones_like(y_ref))
    f.backward(Tensor.ones(y.shape))
    torch.testing.assert_close(x.grad.torch(), x_ref.grad, rtol=tol, atol=tol)

    
