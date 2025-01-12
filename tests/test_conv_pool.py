import pytest
from tensor import Tensor
from MyTorch.nn import Conv2d, MaxPool2d

import torch
from torch.nn import functional as F

tol = 1e-5

def test_conv2d():
    x = Tensor.rand((2, 3, 16, 16), requires_grad=True)
    conv = Conv2d(3, 4)
    y = conv(x)
    assert y.shape == (2, 4, 16, 16)
    assert y.device == x.device
    x_ref = x.torch()
    k_ref = conv.kernel.torch()
    b_ref = conv.bias.torch()
    y_ref = F.conv2d(x_ref, k_ref, b_ref, padding=1)
    torch.testing.assert_close(y.torch(), y_ref, rtol=tol, atol=tol)

    y_ref.backward(torch.ones_like(y_ref))  
    conv.backward(Tensor.ones(y.shape))
    torch.testing.assert_close(x.grad.torch(), x_ref.grad, rtol=tol, atol=tol)
    torch.testing.assert_close(conv.kernel.grad.torch(), k_ref.grad, rtol=tol, atol=tol)
    torch.testing.assert_close(conv.bias.grad.torch(), b_ref.grad, rtol=tol, atol=tol)

def test_pooling2d():
    x = Tensor.rand((2, 3, 16, 16), requires_grad=True)
    pool = MaxPool2d()
    y = pool(x)
    assert y.shape == (2, 3, 8, 8)
    assert y.device == x.device
    x_ref = x.torch()
    y_ref = F.max_pool2d(x_ref, kernel_size=2, stride=2)
    torch.testing.assert_close(y.torch(), y_ref, rtol=tol, atol=tol)

    y_ref.backward(torch.ones_like(y_ref))
    pool.backward(Tensor.ones(y.shape))
    torch.testing.assert_close(x.grad.torch(), x_ref.grad, rtol=tol, atol=tol)