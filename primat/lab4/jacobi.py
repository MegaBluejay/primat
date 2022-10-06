from bisect import bisect_right
from math import sqrt, copysign, atan2, sin, cos, isclose
from functools import cache

import numpy as np

from primat.lab3.csr import Csr


def rotate(a, p, k, l):
    n = len(a)
    aDiff = a[l, l] - a[k, k]
    if isclose(a[l, l], a[k, k], rel_tol=1e-36):
        t = a[k, l] / aDiff
    else:
        phi = aDiff / (2 * a[k, l])
        t = copysign(1, phi) / (abs(phi) + sqrt(phi**2 + 1))
    c = 1 / sqrt(t**2 + 1)
    s = t * c
    tau = s / (1 + c)
    temp = a[k, l]
    a[k, l] = 0.0
    a[k, k] = a[k, k] - t * temp
    a[l, l] = a[l, l] + t * temp
    for i in range(k):  # Case of i < k
        temp = a[i, k]
        a[i, k] = temp - s * (a[i, l] + tau * temp)
        a[i, l] = a[i, l] + s * (temp - tau * a[i, l])
    for i in range(k + 1, l):  # Case of k < i < l
        temp = a[k, i]
        a[k, i] = temp - s * (a[i, l] + tau * a[k, i])
        a[i, l] = a[i, l] + s * (temp - tau * a[i, l])
    for i in range(l + 1, n):  # Case of i > l
        temp = a[k, i]
        a[k, i] = temp - s * (a[l, i] + tau * temp)
        a[l, i] = a[l, i] + s * (temp - tau * a[l, i])
    for i in range(n):  # Update transformation matrix
        temp = p[i, k]
        p[i, k] = temp - s * (p[i, l] + tau * p[i, k])
        p[i, l] = p[i, l] + s * (temp - tau * p[i, l])
    return a, p


def jacobi(a, eps):
    mask = ~np.eye(a.shape[0], dtype=bool)
    p = np.eye(a.shape[0], dtype=float)
    while abs(a[q := np.unravel_index(np.argmax(np.abs(np.where(mask, a, 0))), a.shape)]) >= eps:
        a, p = rotate(a, p, *q)
    return a.diagonal(), p


a = np.array([[4, -30, 60, -35], [-30, 300, -675, 420], [60, -675, 1620, -1050], [-35, 420, -1050, 700]], dtype=float)
print(a)
vals, vecs = jacobi(a, 1e-9)
print(vals)
print(vecs)
print(a @ vecs[:, 0], vecs[:, 0] * vals[0])
