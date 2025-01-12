"""
Author: Tong WU
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Union

from backend import NDArray, Device, fn # underlying CUDA-implemented NDArray and functions
from utils import prod


class Tensor(NDArray):
    """Wrapp on NDArray to provide convenient interfaces and automatic differentiation."""
    grad: Union['Tensor', None] #TODO: implement autodiff

    def __init__(self,
               input: Union[Tuple[int], List, np.ndarray, torch.Tensor, 'Tensor', NDArray],
               device: str = 'gpu',
               requires_grad: bool = False) -> None:
        device = Tensor._str2device(device)
        if isinstance(input, Tuple): 
            assert all(d > 0 for d in input), "tuple should contain only postive integers"
            super().__init__(list(input), device)
        elif isinstance(input, np.ndarray) or isinstance(input, List):
            super().__init__(NDArray.from_array(input, device))
        elif isinstance(input, torch.Tensor):
            super().__init__(NDArray.from_array(input.cpu().numpy(), device))
        elif isinstance(input, Tensor) or isinstance(input, NDArray):
            super().__init__(input)
        else: 
            raise NotImplementedError(f"Unsupported input type: {type(input)}")
        
        if requires_grad:   #TODO: Build autodiff based on Tensor.requires_grad
            self.requires_grad = True
            self.grad = Tensor(self.shape, self.device)
        else:
            self.requires_grad = False
            self.grad = None
    
    @classmethod
    def _str2device(cls, device: str) -> Device:
        assert device in ['cpu', 'gpu'], f"Unknown device: {device}"
        return Device.CPU if device == 'cpu' else Device.GPU
    
    @property
    def shape(self) -> Tuple[int]:
        return tuple(super().shape)
    
    @property
    def size(self) -> int:
        return super().size

    @property
    def device(self) -> str:
        return 'cpu' if super().device == Device.CPU else 'gpu'
    
    @property
    def _device(self) -> Device:
        return super().device
    
    @property
    def ndim(self) -> int:
        return len(self.shape)
    
    def __repr__(self) -> str:
        return self.numpy().__repr__().replace('array', 'MyTensor')
    
    def __str__(self) -> str:
        return self.numpy().__str__()
    
    def numpy(self) -> np.ndarray:
        return np.array(self.tolist()).reshape(self.shape)
    
    def torch(self) -> torch.Tensor:
        return torch.tensor(self.numpy(), requires_grad=self.requires_grad)
    
    def to(self, device: str) -> 'Tensor':
        """Move a tensor to device`device`."""
        return Tensor(super().to(Tensor._str2device(device)))
    
    @property
    def T(self) -> 'Tensor':
        assert self.ndim == 2, "Tensor.T only supports transposing 2D tensors!"
        return Tensor(super().T())
    
    def reshape(self, shape: Tuple[int]) -> 'Tensor':
        count = 0
        shape = list(shape)
        for i in range(len(shape)):
            assert shape[i] > 0 or shape[i] == -1, "shape dimension should be positive or -1"
            if shape[i] == -1: 
                count += 1
                assert count <= 1, "At most one dimension can be inferred"
                shape[i] = self.size // -prod(shape)
        return Tensor(super().reshape(shape))


    def swapaxes(self, axis1: int, axis2: int) -> 'Tensor':
        assert 0 <= axis1 < self.ndim and 0 <= axis2 < self.ndim, "axis out of range"
        return Tensor(super().swap(axis1, axis2))
    
    @classmethod
    def ones(cls,
             shape,
             device: str = 'gpu',
             requires_grad: bool = False) -> 'Tensor':
        """Generate a tensor filled with ones of shape `shape`"""
        return Tensor(NDArray(list(shape), 1., Tensor._str2device(device)),
                      requires_grad=requires_grad)
    
    @classmethod
    def rand(cls, 
             shape: Tuple[int], 
             low: float = 0.,
             high: float = 1.,
             device: str = 'gpu',
             requires_grad = False) -> 'Tensor':
        """Generate random values evenly from [low, high] of shape`shape`

        Args:
            shape (Tuple[int]): 
            low (float, optional):
            high (float, optional):
            device (str, optional): 
            requires_grad (bool, optional):
        Returns:
            Tensor:
        """
        return Tensor(NDArray.rand(list(shape), low, high, Tensor._str2device(device)), 
                      requires_grad=requires_grad)





    

    


        
    
        