import pytest
import sys
sys.path.append('..')
from mytorch import Tensor
from mytorch.nn import Linear

import torch
from torch.nn import functional as F

tol = 1e-5

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
    f.backward(Tensor.ones(y.shape))
    torch.testing.assert_close(x.grad.torch(), x_ref.grad, rtol=tol, atol=tol)
    torch.testing.assert_close(f.weight.grad.torch(), w_ref.grad, rtol=tol, atol=tol)
    torch.testing.assert_close(f.bias.grad.torch(), b_ref.grad, rtol=tol, atol=tol)