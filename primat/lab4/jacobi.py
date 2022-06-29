from bisect import bisect_right, bisect_left
from math import copysign, sqrt, inf

from primat.lab3.csr import Csr


def non_diag(a):
    for i in range(a.n):
        start, end = a.r[i : i + 2]
        for q in range(start, end):
            if a.c[q] == i:
                continue
            yield q, a.v[q]


def next_jac(a):
    q = max(non_diag(a), key=lambda t: abs(t[1]))[0]
    j, k = sorted([bisect_right(a.r, q) - 1, a.c[q]])
    if a[j, j] == a[k, k]:
        sin, cos = 1 / sqrt(2), 1 / sqrt(2)
    else:
        tau = (a[j, j] - a[k, k]) / (2 * a[j, k])
        tan = copysign(1 / (abs(tau) + sqrt(1 + tau**2)), tau)
        cos = 1 / sqrt(1 + tau**2)
        sin = tan * cos

    jj = cos**2 * a[j, j] - 2 * sin * cos * a[j, k] + sin**2 * a[k, k]
    kk = sin**2 * a[j, j] + 2 * sin * cos * a[j, k] + cos**2 * a[k, k]
    jk = (cos**2 - sin**2) * a[j, k] + sin * cos * (a[k, k] - a[j, j])

    v, c, r = [], [], [0]
    for i in range(a.n):
        if i == j:
            for m in range(a.m):
                if m == j:
                    x = jj
                elif m == k:
                    x = jk
                else:
                    x = cos * a[j, m] - sin * a[k, m]
                if not x:
                    continue
                v.append(x)
                c.append(m)
        elif i == k:
            for m in range(a.m):
                if m == j:
                    x = jk
                elif m == k:
                    x = kk
                else:
                    x = sin * a[j, m] + cos * a[k, m]
                if not x:
                    continue
                v.append(x)
                c.append(m)
        else:
            start, end = a.r[i : i + 2]
            xj = cos * a[j, i] - sin * a[k, i]
            xk = sin * a[j, i] + cos * a[k, i]
            for x, m in zip(a.v[start:end], a.c[start:end]):
                if m == j:
                    x = xj
                elif m == k:
                    x = xk
                v.append(x)
                c.append(m)
            qj = bisect_left(c, j, r[-1])
            if qj == len(c) or c[qj] != j:
                c.insert(qj, j)
                v.insert(qj, xj)
            qk = bisect_left(c, k, r[-1])
            if qk == len(c) or c[qk] != k:
                c.insert(qk, k)
                v.insert(qk, xk)
        r.append(len(v))
    return Csr((v, c, r, a.m))


def jacobi(a, eps):
    while abs(max(non_diag(a), key=lambda t: abs(t[1]))[1]) >= eps:
        a = next_jac(a)
    return [a[i, i] for i in range(a.n)]


import numpy as np

a = np.random.uniform(-10, 10, 25).reshape((5, 5))
a = a @ a.T
print(a)
print(np.linalg.eig(a))
print(jacobi(Csr(a), 1))
