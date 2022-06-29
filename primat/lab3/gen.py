from random import choices

import numpy as np

from primat.lab3.csr import *
import math


def gilbert(n):
    A = [[1 / (i + j + 1) for j in range(n)] for i in range(n)]
    x = [(i + 1) for i in range(n)]
    F = np.matmul(np.array(A), np.array(x))
    return Csr(A), F


def qq(n, k):
    q = list(range(-4, 1))
    v, c, r, f = [], [], [0], []
    for i in range(n):
        row = list(choices(q, k=n - 1))
        row.insert(i, -sum(row) + 10 ** (-k))
        fv = 0
        for j, w in enumerate(row):
            if not w:
                continue
            v.append(w)
            c.append(j)
            fv += w * (j + 1)
        f.append(fv)
        r.append(len(v))
    return Csr((v, c, r, n)), f


def almost_diag(n):
    v, c, r, f = [], [], [0], []
    points = []
    for i in range(n):
        val = math.ceil(np.random.random() * 4 + 1)
        points.append((i,i,val))
        fv = 0
        fv += val * (i + 1)
        if np.random.random() > 0.5:
            points.append((i, (i+1) % n, val-1))
            fv += (val-1) * ((i+1) % n + 1)
        f.append(fv)
        r.append(len(v))
    return Csr.from_clist(points,n,n), f
