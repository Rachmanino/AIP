"""
`Tensor` class: wraps NDArray and provides autograd.
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple, List

from backend import NDArray, Device, fn # underlying CUDA-implemented NDArray and functions
from .utils import *

__all__ = ['Tensor', 'Op', 'sqrt', 'log', 'exp']

class Tensor(NDArray):
    """The class wrapping NDArray to provide convenient interfaces and automatic differentiation."""
    op: Optional['Op']
    inputs: List['Tensor']
    requires_grad: bool

    ### Initialization ### 
    def __init__(self,
               input: Union[Tuple[int], List, np.ndarray, torch.Tensor, 'Tensor', NDArray],
               device: str = 'gpu',
               requires_grad: bool = False) -> None:
        device = Tensor._str2device(device)
        if isinstance(input, Tuple): 
            assert all(d > 0 for d in input), "tuple should contain only postive integer"
            super().__init__(list(input), device)
        elif isinstance(input, np.ndarray) or isinstance(input, List):
            super().__init__(NDArray.from_array(input, device))
        elif isinstance(input, torch.Tensor):
            super().__init__(NDArray.from_array(input.cpu().numpy(), device))
        elif isinstance(input, Tensor) or isinstance(input, NDArray):
            super().__init__(input)
        else: 
            raise NotImplementedError(f"Unsupported input type: {type(input)}")
        self.requires_grad = requires_grad
        self.op = None
        self.inputs = []

    
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
    
    def item(self) -> float:
        """Return the value of this tensor as a Python number."""
        assert self.size == 1, "Only scalar tensor can be converted to a Python number!"
        assert self.shape == (1,), "Only scalar tensor can be converted to a Python number!"
        return self.numpy().item()
    
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
        return Transpose()(self)
    
    ### Basic operations ### 
    def detach(self) -> 'Tensor':
        """Detach a tensor from the computation graph."""
        return Tensor(self, requires_grad=self.requires_grad)

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
                shape[i] = self.size // -_prod(shape)
        return Reshape(tuple(shape))(self)


    def swapaxes(self, axis1: int, axis2: int) -> 'Tensor':
        """Swap two axes of a tensor."""
        assert 0 <= axis1 < self.ndim and 0 <= axis2 < self.ndim, "At least one of the axes is out of range!"
        return SwapAxes(axis1, axis2)(self)
    
    ### Const tensor factory ### 
    @classmethod
    def const(cls, 
             shape: Tuple[int], 
             value: float, 
             device: str = 'gpu', 
             requires_grad: bool = False) -> 'Tensor':
        """Generate a tensor filled with const `value` of shape `shape`."""
        return Tensor(NDArray(list(shape), value, Tensor._str2device(device)),
                      requires_grad=requires_grad)
                    
    
    @classmethod
    def zeros(cls,
              shape: Tuple[int],
              device: str = 'gpu',
              requires_grad: bool = False) -> 'Tensor':
        """Generate a tensor filled with 0 of shape `shape`."""
        return cls.const(shape, 0., device, requires_grad)
    
    @classmethod
    def zeros_like(cls,
                   tensor: 'Tensor',
                   device: str = 'gpu',
                   requires_grad: bool = False) -> 'Tensor':
        """Generate a tensor filled with 0 of `tensor`'s shape."""
        return cls.zeros(tensor.shape, device, requires_grad)
    
    @classmethod
    def ones(cls,
             shape,
             device: str = 'gpu',
             requires_grad: bool = False) -> 'Tensor':
        """Generate a tensor filled with 1 of shape `shape`"""
        return cls.const(shape, 1., device, requires_grad)
    
    @classmethod
    def ones_like(cls,
                  tensor: 'Tensor',
                  device: str = 'gpu',
                  requires_grad: bool = False) -> 'Tensor':
        """Generate a tensor filled with 1 of `tensor`'s shape."""
        return cls.ones(tensor.shape, device, requires_grad)
    
    ### Random number generation ### 
    @classmethod
    def rand(cls, 
             shape: Tuple[int] = (1,), 
             low: float = 0.,
             high: float = 1.,
             device: str = 'gpu',
             requires_grad = False) -> 'Tensor':
        """Generate random values evenly sampled from [low, high] of shape`shape`."""
        return Tensor(NDArray.rand(list(shape), low, high, Tensor._str2device(device)), 
                      requires_grad=requires_grad)
    
    @classmethod
    def randn(cls, 
              shape: Tuple[int] = (1,), 
              mean: float = 0., 
              std: float = 1., 
              device: str = 'gpu',
              requires_grad = False) -> 'Tensor':
        """Generate random values sampled from normal distribution with `mean` and `std` of shape `shape`."""
        return Tensor(NDArray.randn(list(shape), mean, std, Tensor._str2device(device)), 
                      requires_grad=requires_grad)
        #! curandn只支持偶数size，这里封装后也支持奇数size

    ### Autograd implementation ### 
    def backward(self, grad: Optional['Tensor'] = None) -> None:
        """Backward pass to compute gradients."""
        assert self.requires_grad, "This tensor does not require gradient!"
        if grad is None:
            grad = Tensor.ones_like(self)
        compute_grad(self, grad)
        
    
    def zero_grad(self) -> None:
        """Zero out the gradient of this tensor."""
        assert self.requires_grad, "This tensor does not require gradient!"
        self.grad = Tensor.zeros_like(self)

    def _is_leaf(self) -> bool:
        """Return True if this tensor is a leaf node."""
        return self.op is None
    
    @staticmethod
    def _make_from_op(op: 'Op', inputs: List['Tensor']):
        cached_data = op.compute(*inputs)
        tensor = Tensor(cached_data)
        tensor._init(op, inputs)
        return tensor
    
    def _init(self, op: 'Op', inputs: List['Tensor']):
        """Initialize a non-leaf tensor with an operation and its inputs."""
        self.op = op
        self.inputs = inputs
        self.cached_data = None
        self.requires_grad = any(input.requires_grad for input in inputs)
    
    ### Overloaded operators w/ autograd ###
    def __eq__(self, other: 'Tensor') -> 'Tensor':
        return NDArray.__eq__(self, other)
    
    def __hash__(self) -> int:
        return hash(id(self))
    
    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        if isinstance(other, Tensor):
            return EwiseAdd()(self, other)
        elif isinstance(other, float):
            return ScalarAdd(other)(self)
        else:
            raise NotImplementedError(f"Unsupported type for addition: {type(other)}")
    __radd__ = __add__
        
    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        if isinstance(other, Tensor):
            return EwiseMul()(self, other)
        elif isinstance(other, float):
            return ScalarMul(other)(self)
        else:
            raise NotImplementedError(f"Unsupported type for multiplication: {type(other)}")
    __rmul__ = __mul__
    
    def __neg__(self) -> 'Tensor':
        return Negate()(self)
    
    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        if isinstance(other, Tensor):
            return EwiseAdd()(self, -other)
        elif isinstance(other, float):
            return ScalarAdd(-other)(self)
        else:
            raise NotImplementedError(f"Unsupported type for subtraction: {type(other)}")
    
    def __rsub__(self, other: Union['Tensor', float]) -> 'Tensor':
        return -(self - other)
    
    def __pow__(self, other: float) -> 'Tensor':
        return ScalarPow(other)(self)
    
    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        if isinstance(other, Tensor):
            return EwiseDiv()(self, other)
        elif isinstance(other, float):
            return ScalarDiv(other)(self)
        else:
            raise NotImplementedError(f"Unsupported type for division: {type(other)}")

    def __rtruediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        if isinstance(other, Tensor):
            return EwiseDiv()(other, self)
        elif isinstance(other, float):
            return other * (self ** -1)
        else:
            raise NotImplementedError(f"Unsupported type for division: {type(other)}")
        
    def sqrt(self) -> 'Tensor':
        return sqrt(self)

    def exp(self) -> 'Tensor':
        return exp(self)
    
    def log(self) -> 'Tensor':
        return log(self)
        
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return matmul(self, other)
        
    
def find_topo_sort(node_list: List[Tensor]) -> List[Tensor]:
    topo_order = []
    visited = set()
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order
    

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    ## 请于此填写你的代码
    if node in visited:
        return
    visited.add(node)
    for input_node in node.inputs:
        topo_sort_dfs(input_node, visited, topo_order)
    topo_order.append(node)

def compute_grad(node: Tensor, grad: Tensor) -> None:
    node2grad = {}
    node2grad[node] = [grad]
    reverse_topo_order = list(reversed(find_topo_sort([node])))

    for node in reverse_topo_order:
        node.grad = _sum(node2grad[node])
        if node.op is not None:
            for input_node, grad in zip(node.inputs, node.op.grad_as_tuple(node.grad, node)):
                if input_node not in node2grad:
                    node2grad[input_node] = []
                node2grad[input_node].append(grad)


class Op():
    """The base class for all operations."""
    def __call__(self, *args):
        return Tensor._make_from_op(self, args)
    
    def grad_as_tuple(self, grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        output = self.gradient(grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)
   
class EwiseAdd(Op):
    """Element-wise addition operation."""
    def compute(self, x: NDArray, y: NDArray):
        return NDArray.__add__(x, y)
    
    def gradient(self, grad: Tensor, node: Tensor):
        return grad, grad
    
class EwiseMul(Op):
    """Element-wise multiplication operation."""
    def compute(self, x: NDArray, y: NDArray) -> NDArray:
        return NDArray.__mul__(x, y)
    
    def gradient(self, grad: Tensor, node: Tensor):
        return grad * node.inputs[1], grad * node.inputs[0]
    
class ScalarAdd(Op):
    """Scalar addition operation."""
    def __init__ (self, scalar: float):
        self.scalar = scalar

    def compute(self, x: NDArray) -> NDArray:
        return NDArray.__add__(x, self.scalar)
        
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        return grad
    
class ScalarMul(Op):
    """Scalar multiplication operation."""
    def __init__ (self, scalar: float):
        self.scalar = scalar

    def compute(self, x: NDArray) -> NDArray:
        return NDArray.__mul__(x, self.scalar)
        
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        return grad * self.scalar
    
class Negate(Op):
    """Negation operation."""
    def compute(self, x: NDArray) -> NDArray:
        return NDArray.__mul__(x, -1)
    
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        return -grad
    
class EwiseDiv(Op):
    """Element-wise division operation."""
    def compute(self, x: NDArray, y: NDArray) -> NDArray:
        return NDArray.__truediv__(x, y)
    
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        return grad / node.inputs[1], -grad * node.inputs[0] / node.inputs[1] ** 2
    
class ScalarDiv(Op):
    """Scalar division operation."""
    def __init__ (self, scalar: float):
        self.scalar = scalar

    def compute(self, x: NDArray) -> NDArray:
        return NDArray.__truediv__(x, self.scalar)
        
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        return grad / self.scalar
    
class ScalarPow(Op):
    """Scalar power operation."""
    def __init__ (self, scalar: float):
        self.scalar = scalar

    def compute(self, x: NDArray) -> NDArray:
        return NDArray.__pow__(x, self.scalar)
        
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        return grad * self.scalar * node.inputs[0] ** (self.scalar - 1)
def sqrt(x: Tensor) -> Tensor:
    return ScalarPow(0.5)(x)
    
class Log(Op):
    """Logarithm operation."""
    def compute(self, x: NDArray) -> NDArray:
        return NDArray.log(x)
        
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        return grad / node.inputs[0]
def log(x: Tensor) -> Tensor:
    return Log()(x)
    
class Exp(Op):
    """Exponential operation."""
    def compute(self, x: NDArray) -> NDArray:
        return NDArray.exp(x)
        
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        return grad * node
def exp(x: Tensor) -> Tensor:
    return Exp()(x)

class Transpose(Op):
    """Transpose operation."""
    def compute(self, x: NDArray) -> NDArray:
        return NDArray.T(x)
        
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        return grad.T()

class Reshape(Op):
    """Reshape operation."""
    def __init__ (self, shape: Tuple[int]):
        self.shape = shape

    def compute(self, x: NDArray) -> NDArray:
        return NDArray.reshape(x, self.shape)
        
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        return grad.reshape(node.inputs[0].shape)

class SwapAxes(Op):
    def __init__ (self, axis1: int, axis2: int):
        self.axis1 = axis1
        self.axis2 = axis2
    
    def compute(self, x: NDArray) -> NDArray:
        return NDArray.swap(x, self.axis1, self.axis2)
    
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        return grad.swap(self.axis1, self.axis2)
    
class Matmul(Op):
    """Matrix multiplication operation."""
    def compute(self, x: NDArray, y: NDArray) -> NDArray:
        return NDArray.__matmul__(x, y)
    
    def gradient(self, grad: Tensor, node: Tensor) -> Tensor:
        return grad @ node.inputs[1].T, node.inputs[0].T @ grad
def matmul(x: Tensor, y: Tensor) -> Tensor:
    assert x.ndim == 2 and y.ndim == 2, "Matrix multiplication requires 2D tensors!"
    assert x.shape[1] == y.shape[0], "Matrix multiplication requires the inner dimensions to match!"
    return Matmul()(x, y)

#TODO: summation. max, min, broadcast
    


    
    

    
    
    


    
    


    



    

    


        
    
        