import numpy as np

from primat.lab2.grad import norm_sq


def zeydel(a, b, eps):
    n = a.shape[0]
    xp = np.zeros(b.shape)
    x = b.copy()
    while norm_sq(x - xp) > eps**2:
        xp = x.copy()
        for i in range(n):
            x[i] = sum(-a[i, j] * x[j] / a[i, i] if i != j else 0 for j in range(n))
    return x
