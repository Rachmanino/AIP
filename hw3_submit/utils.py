from functools import reduce

def prod(iterable):
    return reduce(lambda x, y: x * y, iterable, 1)