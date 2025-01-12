"""
Implementation of Tensor class, which wraps NDArray.
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple, List

from backend import NDArray, Device, fn # underlying CUDA-implemented NDArray and functions
from mytorch.utils import *


class Tensor(NDArray):
    """The class wrapping NDArray to provide convenient interfaces and automatic differentiation."""
    grad: Union['Tensor', None] #TODO: implement autodiff

    ### Initialization ### 
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
    
    ### Basic properties ### 
    @classmethod
    def _str2device(cls, device: str) -> Device:
        """Convert string to Device enum."""
        assert device in ['cpu', 'gpu'], f"Unknown device: {device}"
        return Device.CPU if device == 'cpu' else Device.GPU
    
    @property
    def shape(self) -> Tuple[int]:
        """Return the shape of the tensor."""
        return tuple(super().shape)
    
    @property
    def size(self) -> int:
        """Return # elements in the tensor."""
        return super().size

    @property
    def device(self) -> str:
        """Return the device of the tensor, either 'cpu' or 'gpu'."""
        return 'cpu' if super().device == Device.CPU else 'gpu'
    
    @property
    def _device(self) -> Device:
        """Return the device of the tensor's NDArray, either `Device.CPU` or `Device.GPU`."""
        return super().device
    
    @property
    def ndim(self) -> int:
        """Return the number of dimensions of the tensor."""
        return len(self.shape)
    
    def __repr__(self) -> str:
        return self.numpy().__repr__().replace('array', 'MyTensor')
    
    def __str__(self) -> str:
        return self.numpy().__str__()
    
    ### Switch formats ### 
    def numpy(self) -> np.ndarray:
        """Convert a tensor to numpy.ndarray"""
        return np.array(self.tolist()).reshape(self.shape)
    
    def torch(self) -> torch.Tensor:
        """Convert a tensor to torch.Tensor"""
        return torch.tensor(self.numpy(), requires_grad=self.requires_grad)
    
    ### Move between devices ### 
    def to(self, device: str) -> 'Tensor':
        """Move a tensor to device`device`."""
        return Tensor(super().to(Tensor._str2device(device)))

    def cpu(self) -> 'Tensor':
        """Move a tensor to CPU."""
        return self.to('cpu')
    
    def gpu(self) -> 'Tensor':
        """Move a tensor to GPU."""
        return self.to('gpu')
    
    @property
    def T(self) -> 'Tensor':
        """Return the transpose of a 2D tensor."""
        assert self.ndim == 2, "Tensor.T only supports transposing 2D tensors!"
        return Tensor(super().T())
    
    ### Basic operations ### 
    def detach(self) -> 'Tensor':
        """Detach a tensor from the computation graph."""
        return Tensor(self, requires_grad=False)

    def reshape(self, shape: Tuple[int]) -> 'Tensor':
        """
        Reshape the tensor to the given shape.
        """
        count = 0
        shape = list(shape)
        for i in range(len(shape)):
            assert shape[i] > 0 or shape[i] == -1, "shape dimension should be positive or -1!"
            if shape[i] == -1: 
                count += 1
                assert count <= 1, "At most one dimension can be inferred"
                shape[i] = self.size // -prod(shape)
        return Tensor(super().reshape(shape))


    def swapaxes(self, axis1: int, axis2: int) -> 'Tensor':
        """Swap two axes of a tensor."""
        assert 0 <= axis1 < self.ndim and 0 <= axis2 < self.ndim, "At least one of the axes is out of range!"
        return Tensor(super().swap(axis1, axis2))
    
    ### Fill functions ### 
    @classmethod
    def fill(cls, 
             shape: Tuple[int], 
             value: float, 
             device: str = 'gpu', 
             requires_grad: bool = False) -> 'Tensor':
        """Generate a tensor filled with `value` of shape `shape`."""
        return Tensor(NDArray(list(shape), value, Tensor._str2device(device)),
                      requires_grad=requires_grad)
    
    @classmethod
    def zeros(cls,
              shape: Tuple[int],
              device: str = 'gpu',
              requires_grad: bool = False) -> 'Tensor':
        """Generate a tensor filled with 0 of shape `shape`."""
        return cls.fill(shape, 0., device, requires_grad)
    
    @classmethod
    def ones(cls,
             shape,
             device: str = 'gpu',
             requires_grad: bool = False) -> 'Tensor':
        """Generate a tensor filled with 1 of shape `shape`"""
        return cls.fill(shape, 1., device, requires_grad)
    
    ### Random number generation ### 
    @classmethod
    def rand(cls, 
             shape: Tuple[int], 
             low: float = 0.,
             high: float = 1.,
             device: str = 'gpu',
             requires_grad = False) -> 'Tensor':
        """Generate random values evenly sampled from [low, high] of shape`shape`."""
        return Tensor(NDArray.rand(list(shape), low, high, Tensor._str2device(device)), 
                      requires_grad=requires_grad)
    
    @classmethod
    def randn(cls, 
              shape: Tuple[int], 
              mean: float = 0., 
              std: float = 1., 
              device: str = 'gpu',
              requires_grad = False) -> 'Tensor':
        """Generate random values sampled from normal distribution with `mean` and `std` of shape `shape`."""
        return Tensor(NDArray.randn(list(shape), mean, std, Tensor._str2device(device)), 
                      requires_grad=requires_grad)
        #! curandn只支持偶数size，这里封装后也支持奇数size
    






    

    


        
    
        