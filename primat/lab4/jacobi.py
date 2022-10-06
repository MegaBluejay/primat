from itertools import chain, groupby
from math import sqrt, copysign, isclose
from operator import itemgetter as ig

import numpy as np
from toolz import merge_sorted

from primat.lab3.csr import Csr


def mod_row(a, ri, new_a, mods):
    start, end = a.r[ri : ri + 2]
    ogs = zip(a.c[start:end], a.v[start:end])
    q = list(merge_sorted(ogs, mods, key=ig(0)))
    for i in range(len(q)):
        c, v = q[i]
        if (i + 1 < len(q) and q[i + 1][0] == c) or v == 0:
            continue
        new_a.c.append(c)
        new_a.v.append(v)
    new_a.r.append(len(new_a.c))


def compile_mods(*mods):
    return {i: list(map(ig(1), row)) for i, row in groupby(sorted(chain.from_iterable(mods)), key=ig(0))}


def rotate(a, p, k, l):
    print(a.to_mat())
    print(p.to_mat())
    if isclose(a[l, l], a[k, k], rel_tol=1e-36):
        t = a[k, l] / (a[l, l] - a[k, k])
    else:
        phi = (a[l, l] - a[k, k]) / (2 * a[k, l])
        t = copysign(1, phi) / (abs(phi) + sqrt(phi**2 + 1))
    c = 1 / sqrt(t**2 + 1)
    s = t * c
    tau = s / (1 + c)
    new_a = Csr(([], [], [0], a.n))
    a_mods = compile_mods(
        [
            (k, (l, 0)),
            (k, (k, a[k, k] - t * a[k, l])),
            (l, (l, a[l, l] + t * a[k, l])),
        ],
        ((i, (k, a[i, k] - s * (a[i, l] + tau * a[i, k]))) for i in range(k)),
        ((i, (l, a[i, l] + s * (a[i, k] - tau * a[i, l]))) for i in range(k)),
        ((k, (i, a[k, i] - s * (a[i, l] + tau * a[k, i]))) for i in range(k + 1, l)),
        ((i, (l, a[i, l] + s * (a[k, i] - tau * a[i, l]))) for i in range(k + 1, l)),
        ((k, (i, a[k, i] - s * (a[l, i] + tau * a[k, i]))) for i in range(l + 1, a.n)),
        ((l, (i, a[l, i] + s * (a[k, i] - tau * a[l, i]))) for i in range(l + 1, a.n)),
    )
    for i in range(a.n):
        mod_row(a, i, new_a, a_mods.get(i, []))
    p_mods = compile_mods(
        ((i, (k, p[i, k] - s * (p[i, l] + tau * p[i, k]))) for i in range(a.n)),
        ((i, (l, p[i, l] + s * (p[i, k] - tau * p[i, l]))) for i in range(a.n)),
    )
    new_p = Csr(([], [], [0], a.n))
    for i in range(a.n):
        mod_row(p, i, new_p, p_mods.get(i, []))
    return new_a, new_p


# def rotate(a, p, k, l):
#     n = len(a)
#     aDiff = a[l, l] - a[k, k]
#     if isclose(a[l, l], a[k, k], rel_tol=1e-36):
#         t = a[k, l] / aDiff
#     else:
#         phi = aDiff / (2 * a[k, l])
#         t = copysign(1, phi) / (abs(phi) + sqrt(phi**2 + 1))
#     c = 1 / sqrt(t**2 + 1)
#     s = t * c
#     tau = s / (1 + c)
#     temp = a[k, l]
#     a[k, l] = 0.0
#     a[k, k] = a[k, k] - t * temp
#     a[l, l] = a[l, l] + t * temp
#     for i in range(k):  # Case of i < k
#         temp = a[i, k]
#         a[i, k] = temp - s * (a[i, l] + tau * temp)
#         a[i, l] = a[i, l] + s * (temp - tau * a[i, l])
#     for i in range(k + 1, l):  # Case of k < i < l
#         temp = a[k, i]
#         a[k, i] = temp - s * (a[i, l] + tau * a[k, i])
#         a[i, l] = a[i, l] + s * (temp - tau * a[i, l])
#     for i in range(l + 1, n):  # Case of i > l
#         temp = a[k, i]
#         a[k, i] = a[k, i] - s * (a[l, i] + tau * a[k, i])
#         a[l, i] = a[l, i] + s * (temp - tau * a[l, i])
#     for i in range(n):  # Update transformation matrix
#         temp = p[i, k]
#         p[i, k] = temp - s * (p[i, l] + tau * p[i, k])
#         p[i, l] = p[i, l] + s * (temp - tau * p[i, l])
#     return a, p


def non_diag(a):
    for i in range(a.n):
        start, end = a.r[i : i + 2]
        for j in range(start, end):
            if a.c[j] != i:
                yield i, a.c[j], a.v[j]


def jacobi(a, eps):
    p = Csr(([1] * a.n, list(range(a.n)), list(range(a.n + 1)), a.n))
    while abs((q := max(non_diag(a), key=lambda w: abs(w[2])))[2]) >= eps:
        a, p = rotate(a, p, *q[:2])
    return [a[i, i] for i in range(a.n)], p


# def jacobi(a, eps):
#     mask = ~np.eye(a.shape[0], dtype=bool)
#     p = np.eye(a.shape[0], dtype=float)
#     while abs(a[q := np.unravel_index(np.argmax(np.abs(np.where(mask, a, 0))), a.shape)]) >= eps:
#         a, p = rotate(a, p, *sorted(q))
#     return a.diagonal(), p


a = np.array(
    [
        [4, -30, 60, -35],
        [-30, 300, -675, 420],
        [60, -675, 1620, -1050],
        [-35, 420, -1050, 700],
    ],
    dtype=float,
)
print(a)
vals, vecs = jacobi(Csr(a), 1e-9)
vecs = np.array(vecs.to_mat())
print(a @ vecs[:, 0], vecs[:, 0] * vals[0])
