import numpy as np

from primat.lab2.grad import norm_sq
from primat.lab3.csr import Csr


def zeydel(a, b, eps):
    n = a.n
    xp = np.zeros(b.shape)
    x = b.copy()
    while norm_sq(x - xp) > eps**2:
        xp = x.copy()
        for i in range(n):
            x[i] = b[i]
            start, end = a.r[i : i + 2]
            for v, j in zip(a.v[start:end], a.c[start:end]):
                if i != j:
                    x[i] -= v * x[j]
            x[i] /= a[i, i]
    return x


a = Csr([[1, 2], [3, 4]])
b = np.array([1, 1])
print(zeydel(a, b, 1e-5))
