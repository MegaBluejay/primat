import numpy as np

from primat.lab2.grad import norm_sq
from primat.lab3.csr import Csr


def zeydel(a, b, eps, om):
    n = a.n
    xp = np.zeros(b.shape)
    x = b.copy()
    while norm_sq(x - xp) > eps**2:
        xp = x.copy()
        for i in range(n):
            x[i] *= 1 - om
            q = 0
            start, end = a.r[i : i + 2]
            for v, j in zip(a.v[start:end], a.c[start:end]):
                if i != j:
                    q += v * x[j]
            x[i] += om * (b[i] - q) / a[i, i]
    return x


a = Csr([[1, 2], [3, 4]])
b = np.array([1, 1])
print(zeydel(a, b, 1e-5, 1.1))
