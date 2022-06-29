from random import choices

from primat.lab3.csr import *


def gilbert(n):
    return [[1 / (i + j + 1) for j in range(n)] for i in range(n)]


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
