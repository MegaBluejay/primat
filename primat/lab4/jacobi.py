from bisect import bisect_left
from itertools import chain, groupby
from math import sqrt, copysign, isclose
from operator import itemgetter as ig
from collections import namedtuple

from toolz import merge_sorted

from primat.lab3.csr import Csr


JacobiResult = namedtuple("JacobiResult", ["vals", "vecs", "its"])


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
    if isclose(a[l, l], a[k, k]):
        t = 1
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


def top_tri(a):
    for i in range(a.n):
        start, end = a.r[i : i + 2]
        start = bisect_left(a.c, i + 1, start, end)
        for j in range(start, end):
            yield i, a.c[j], a.v[j]


def jacobi(a, eps=None, maxit=None):
    p = Csr(([1] * a.n, list(range(a.n)), list(range(a.n + 1)), a.n))
    i = 0
    while True:
        q = max(top_tri(a), key=lambda w: abs(w[2]))
        if (eps is not None and abs(q[2]) < eps) or (maxit is not None and i == maxit):
            break
        a, p = rotate(a, p, *q[:2])
        i += 1
    return JacobiResult([a[i, i] for i in range(a.n)], p, i)
