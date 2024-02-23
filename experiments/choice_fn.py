import math
import random


def heat_chooser(*args, **kwargs):
    if not args:
        raise TypeError('min expected 1 arguments, got 0')
    elif len(args) == 1:
        iterable = iter(args[0])
    else:
        iterable = args
    key = None
    for k, v in kwargs.items():
        if k != 'key':
            raise TypeError('min() got an unexpected keyword argument')
        elif not callable(v):
            raise TypeError(str(type(v).__name__) + ' object is not callable')
        key = v
    iterable = list(iterable)
    costs = list(map(key, iterable))
    min_cost = min(costs)
    weights = list(map(lambda x: math.exp(-x+min_cost), costs))
    if sum(weights):
        weights = [float(i)/sum(weights) for i in weights]
    return random.choices(iterable, weights=weights, k=1)[0]


def random_chooser(*args, **kwargs):
    if not args:
        raise TypeError('min expected 1 arguments, got 0')
    elif len(args) == 1:
        iterable = iter(args[0])
    else:
        iterable = args
    iterable = list(iterable)
    return random.choice(iterable)


if __name__ == '__main__':
    weights = [(1, 1, 20), (2, 2, 4), (3, 3, 5)]
    print(random_chooser(weights, key=lambda x: x[2]))
    print(heat_chooser(weights, key=lambda x: x[2]))
