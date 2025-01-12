import pytest
import os
from tensor import Tensor
from MyTorch.nn import CELoss

import torch
from torch.nn import functional as F

tol = 1e-5

def test_softmax_celoss():
    f = CELoss()
    logits = Tensor.rand((4, 10), requires_grad=True)
    logits_ref = logits.torch()
    labels_ref = torch.randint(0, 10, (4,))
    labels = Tensor(labels_ref.numpy())
    loss = f(logits, labels) # reduction is 'mean', loss is scalar(fp32)
    loss_ref = F.cross_entropy(logits_ref, labels_ref, reduction='mean')
    torch.testing.assert_close(loss, loss_ref.item(), rtol=tol, atol=tol)

    loss_ref.backward()
    f.backward()
    torch.testing.assert_close(logits.grad.torch(), logits_ref.grad, rtol=tol, atol=tol)


