__all__ = ["weight_set_generator","generate_alternating_range","divide_by_gcd_matrix"]

from itertools import product
import math

import numpy as np
import numpy.typing as npt


def weight_set_generator(kappa: int, n: int):
    a = list(product(range(-kappa, kappa + 1), repeat=n))
    for i in a:
        yield i


def generate_alternating_range(n: int, start_with_zero: bool = True) -> list[int]:
    sequence = [0] if start_with_zero else []
    for i in range(1, n + 1):
        sequence.append(i)
        sequence.append(-i)
    return sequence


def divide_by_gcd_matrix(a: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    n_rows, n_cols = a.shape
    b = np.zeros_like(a)
    for i in range(n_cols):
        b[:,i] = - a[:,i] / math.gcd(*a[:,i])
    return b
