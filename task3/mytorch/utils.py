from functools import reduce
from operator import add, mul

__all__ = ['_prod', '_sum', '_unique',
           ]

def _prod(iterable):
    return reduce(mul, iterable, 1)

def _sum(iterable):
    return reduce(add, iterable)

def _unique(iterable):
    return list(set(iterable))




    