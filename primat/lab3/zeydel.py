import numpy as np

from primat.lab2.grad import norm_sq
from primat.lab3.csr import Csr


def zeydel(a, b, eps):
    counter = 0
    n = a.n
    xp = None
    x = b.copy()
    while xp is None or norm_sq(x - xp) > eps**2:
        counter += 1
        xp = x.copy()
        for i in range(n):
            x[i] = b[i]
            start, end = a.r[i : i + 2]
            for v, j in zip(a.v[start:end], a.c[start:end]):
                if i != j:
                    x[i] -= v * x[j]
            x[i] /= a[i, i]
    return x, counter
