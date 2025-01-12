from functools import reduce

__all__ = ['prod', 'sum', 'max', 'min', 'unique']

def prod(iterable):
    return reduce(lambda x, y: x * y, iterable, 1)

def sum(iterable):
    return reduce(lambda x, y: x + y, iterable, 0)

def max(iterable):
    return reduce(lambda x, y: x if x > y else y, iterable, float('-inf'))

def min(iterable):
    return reduce(lambda x, y: x if x < y else y, iterable, float('inf'))

def unique(iterable):
    return list(set(iterable))




    