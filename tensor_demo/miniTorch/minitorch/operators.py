"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    "$f(x, y) = x * y$"
    # ASSIGN0.1
    return x * y
    # END ASSIGN0.1


def id(x: float) -> float:
    "$f(x) = x$"
    # ASSIGN0.1
    return x
    # END ASSIGN0.1


def add(x: float, y: float) -> float:
    "$f(x, y) = x + y$"
    # ASSIGN0.1
    return x + y
    # END ASSIGN0.1


def neg(x: float) -> float:
    "$f(x) = -x$"
    # ASSIGN0.1
    return -x
    # END ASSIGN0.1


def lt(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is less than y else 0.0"
    # ASSIGN0.1
    return 1.0 if x < y else 0.0
    # END ASSIGN0.1


def eq(x: float, y: float) -> float:
    "$f(x) =$ 1.0 if x is equal to y else 0.0"
    # ASSIGN0.1
    return 1.0 if x == y else 0.0
    # END ASSIGN0.1


def max(x: float, y: float) -> float:
    "$f(x) =$ x if x is greater than y else y"
    # ASSIGN0.1
    return x if x > y else y
    # END ASSIGN0.1


def is_close(x: float, y: float) -> float:
    "$f(x) = |x - y| < 1e-2$"
    # ASSIGN0.1
    return (x - y < 1e-2) and (y - x < 1e-2)
    # END ASSIGN0.1


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    # ASSIGN0.1
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))
    # END ASSIGN0.1


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    # ASSIGN0.1
    return x if x > 0 else 0.0
    # END ASSIGN0.1


EPS = 1e-6


def log(x: float) -> float:
    "$f(x) = log(x)$"
    return math.log(x + EPS)


def exp(x: float) -> float:
    "$f(x) = e^{x}$"
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    r"If $f = log$ as above, compute $d \times f'(x)$"
    # ASSIGN0.1
    return d / (x + EPS)
    # END ASSIGN0.1


def inv(x: float) -> float:
    "$f(x) = 1/x$"
    # ASSIGN0.1
    return 1.0 / x
    # END ASSIGN0.1


def inv_back(x: float, d: float) -> float:
    r"If $f(x) = 1/x$ compute $d \times f'(x)$"
    # ASSIGN0.1
    return -(1.0 / x**2) * d
    # END ASSIGN0.1


def relu_back(x: float, d: float) -> float:
    r"If $f = relu$ compute $d \times f'(x)$"
    # ASSIGN0.1
    return d if x > 0 else 0.0
    # END ASSIGN0.1


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
         A function that takes a list, applies `fn` to each element, and returns a
         new list
    """
    # ASSIGN0.3
    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map
    # END ASSIGN0.3


def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    # ASSIGN0.3
    return map(neg)(ls)
    # END ASSIGN0.3


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
         Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """
    # ASSIGN0.3
    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith
    # END ASSIGN0.3


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    # ASSIGN0.3
    return zipWith(add)(ls1, ls2)
    # END ASSIGN0.3


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
         Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """
    # ASSIGN0.3
    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce
    # END ASSIGN0.3


def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    # ASSIGN0.3
    return reduce(add, 0.0)(ls)
    # END ASSIGN0.3


def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    # ASSIGN0.3
    return reduce(mul, 1.0)(ls)
    # END ASSIGN0.3
