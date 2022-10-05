from bisect import bisect_right
from math import sqrt, copysign, atan2, sin, cos, isclose
from functools import cache

import numpy as np

from primat.lab3.csr import Csr


def rotate(a, p, j, k):
    if isclose(a[j, j], a[k, k], rel_tol=1e-20):
        s, c = [1 / sqrt(2)] * 2
        tau = np.pi / 4
    else:
        tau = (a[j, j] - a[k, k]) / (2 * a[j, k])
        t = copysign(1, tau) / (abs(tau) + sqrt(1 + tau**2))
        c = 1 / sqrt(1 + t**2)
        s = c * t
    new = np.copy(a)
    new[j, j] = c**2 * a[j, j] - 2 * s * c * a[j, k] + s**2 * a[k, k]
    new[k, k] = s**2 * a[j, j] + 2 * s * c * a[j, k] + c**2 * a[k, k]
    new[j, k] = new[k, j] = (c**2 - s**2) * a[j, k] + s * c * (a[k, k] - a[j, j])
    for m in range(a.shape[0]):
        if m in [j, k]:
            continue
        new[j, m] = new[m, j] = c * a[j, m] - s * a[k, m]
        new[k, m] = new[m, k] = s * a[j, m] + c * a[k, m]
    for i in range(a.shape[0]):  # Update transformation matrix
        temp = p[i, j]
        p[i, j] = temp - s * (p[i, k] + tau * p[i, j])
        p[i, k] = p[i, k] + s * (temp - tau * p[i, k])
    return new, p


# def jacobi(a, eps):
#     while abs(max(non_diag(a), key=lambda t: abs(t[1]), default=(-1, 0))[1]) >= eps:
#         a = rotate(a)
#     return [a[i, i] for i in range(a.n)]


def jacobi(a, eps):
    mask = ~np.eye(a.shape[0], dtype=bool)
    p = np.eye(a.shape[0], dtype=float)
    while abs(a[q := np.unravel_index(np.argmax(np.abs(np.where(mask, a, 0))), a.shape)]) >= eps:
        a, p = rotate(a, p, *q)
    print(a)
    return a.diagonal(), p


a = np.array([[4, -30, 60, -35], [-30, 300, -675, 420], [60, -675, 1620, -1050], [-35, 420, -1050, 700]], dtype=float)
print(a)
vals, vecs = jacobi(a, 1e-9)
print(vals)
print(vecs)
print(a @ vecs[:, 0], vecs[:, 0] * vals[0])
