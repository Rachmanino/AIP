import pytest
import sys
sys.path.append('..')
from mytorch import *
from mytorch.nn import *

tol = 1e-5

def test_algebra():
    a = Tensor([[1, 4], [9, 16]])
    b = Tensor([[2, 2], [2, 2]])
    # test eq
    assert a == a
    assert a != b

    # test algebra
    assert a + 1. == Tensor([[2, 5], [10, 17]])
    assert a + b == Tensor([[3, 6], [11, 18]])

    assert a - 1. == Tensor([[0, 3], [8, 15]])
    assert a - b == Tensor([[-1, 2], [7, 14]])

    assert a * 2. == Tensor([[2, 8], [18, 32]])
    assert a * b == Tensor([[2, 8], [18, 32]])

    assert a / 2. == Tensor([[0.5, 2], [4.5, 8]])
    assert a / b == Tensor([[0.5, 2], [4.5, 8]])

    assert -a == Tensor([[-1, -4], [-9, -16]])
    
    assert a ** 2. == Tensor([[1, 16], [81, 256]])
    assert sqrt(a) == Tensor([[1, 2], [3, 4]])

    # test reshape
    assert a.reshape((1, 4)) == Tensor([[1, 4, 9, 16]])
    assert a.T == Tensor([[1, 9], [4, 16]])
    assert a.swapaxes(0, 1) == Tensor([[1, 9], [4, 16]])

def test_autograd():
    # test backward
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = x + 1.
    output = x ** 2. + x @ y
    output.backward()
    assert x.grad == Tensor([[11, 17], [17, 23]])

    # test detach method
    x = x.detach()
    assert x == Tensor([[1, 2], [3, 4]])
    assert x.requires_grad and x._is_leaf()

