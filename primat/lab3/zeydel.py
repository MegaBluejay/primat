import numpy as np

from primat.lab2.grad import norm_sq


def zeydel(a, b, eps):
    n = a.n
    xp = np.zeros(b.shape)
    x = b.copy()
    while norm_sq(x - xp) > eps**2:
        xp = x.copy()
        for i in range(n):
            x[i] = (b[i] - sum(a[i, j] * x[j] if i != j else 0 for j in range(n))) / a[i, i]
    return x
