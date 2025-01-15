import pytest
import sys
from mytorch import Tensor
from mytorch.nn import *

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
    y.backward(Tensor.ones(y.shape))
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
    assert y.requires_grad
    y.backward()
    torch.testing.assert_close(x.grad.torch(), x_ref.grad, rtol=tol, atol=tol)

def test_fc():
    f = Linear(4, 3)
    x = Tensor.rand((2,4), requires_grad=True)
    y = f(x)
    assert y.shape == (2,3)
    assert y.device == x.device
    x_ref = x.torch()
    w_ref = f.weight.torch()
    b_ref = f.bias.torch()
    y_ref = F.linear(x_ref, w_ref.T, b_ref) # F.linear() need to pass in w.T
    assert y_ref.requires_grad
    torch.testing.assert_close(y.torch(), y_ref, rtol=tol, atol=tol)

    y_ref.backward(torch.ones_like(y_ref))
    y.backward()
    torch.testing.assert_close(x.grad.torch(), x_ref.grad, rtol=tol, atol=tol)
    torch.testing.assert_close(f.weight.grad.torch(), w_ref.grad, rtol=tol, atol=tol)
    torch.testing.assert_close(f.bias.grad.torch(), b_ref.grad, rtol=tol, atol=tol)

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
    y.backward()
    torch.testing.assert_close(x.grad.torch(), x_ref.grad, rtol=tol, atol=tol)
    torch.testing.assert_close(conv.kernel.grad.torch(), k_ref.grad, rtol=tol, atol=tol)
    torch.testing.assert_close(conv.bias.grad.torch(), b_ref.grad, rtol=tol, atol=tol)

def test_pool2d():
    x = Tensor.rand((2, 3, 16, 16), requires_grad=True)
    pool = MaxPool2d(2, 2)
    y = pool(x)
    assert y.shape == (2, 3, 8, 8)
    assert y.device == x.device
    x_ref = x.torch()
    y_ref = F.max_pool2d(x_ref, kernel_size=2, stride=2)
    torch.testing.assert_close(y.torch(), y_ref, rtol=tol, atol=tol)

    y_ref.backward(torch.ones_like(y_ref))
    y.backward()
    torch.testing.assert_close(x.grad.torch(), x_ref.grad, rtol=tol, atol=tol)

def test_softmax_celoss():
    f = CELoss()
    logits = Tensor.rand((4, 10), requires_grad=True)
    logits_ref = logits.torch()
    labels_ref = torch.randint(0, 10, (4,))
    labels = Tensor(labels_ref.numpy())
    loss = f(logits, labels) # reduction is 'mean', loss is scalar(fp32)
    loss_ref = F.cross_entropy(logits_ref, labels_ref, reduction='mean')
    torch.testing.assert_close(loss.item(), loss_ref.item(), rtol=tol, atol=tol)

    loss_ref.backward()
    loss.backward()
    torch.testing.assert_close(logits.grad.torch(), logits_ref.grad, rtol=tol, atol=tol)


