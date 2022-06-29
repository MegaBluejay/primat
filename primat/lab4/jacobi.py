from bisect import bisect_right, bisect_left
from math import copysign, sqrt, inf, atan

from primat.lab3.csr import Csr


def non_diag(a):
    for i in range(a.n):
        start, end = a.r[i : i + 2]
        for q in range(start, end):
            if a.c[q] == i:
                continue
            yield q, a.v[q]


def next_jac(a):
    q = max(non_diag(a), key=lambda t: abs(t[1]), default=(-1, 0))[0]
    j, k = sorted([bisect_right(a.r, q) - 1, a.c[q]])
    if a[j, j] == a[k, k]:
        s, c = 1 / sqrt(2), 1 / sqrt(2)
    else:
        tau = (a[j, j] - a[k, k]) / (2 * a[j, k])
        c = 1 / sqrt(1 + tau**2)
        s = copysign(1 / (abs(tau) + sqrt(1 + tau**2)), tau) * c

    b = Csr(([], [], [0], a.n))
    for i in range(a.n):
        for m in range(i):
            x = b[m, i]
            if x:
                b.v.append(x)
                b.c.append(m)
        for m in range(i, a.n):
            if [i, m] == [j, j]:
                x = c**2 * a[j, j] - 2 * s * c * a[j, k] + s**2 * a[k, k]
            elif [i, m] == [k, k]:
                x = s**2 * a[j, j] + 2 * s * c * a[j, k] + c**2 * a[k, k]
            elif [i, m] == [j, k]:
                x = (c**2 - s**2) * a[j, k] + s * c * (a[k, k] - a[j, j])
            elif j in [i, m]:
                x = c * a[j, m] - s * a[k, m]
            elif k in [i, m]:
                x = s * a[j, m] + c * a[k, m]
            else:
                x = a[i, m]
            if x:
                b.v.append(x)
                b.c.append(m)
        b.r.append(len(b.v))
    return b


def jacobi(a, eps):
    while abs(max(non_diag(a), key=lambda t: abs(t[1]), default=(-1, 0))[1]) >= eps:
        a = next_jac(a)
    return [a[i, i] for i in range(a.n)]
