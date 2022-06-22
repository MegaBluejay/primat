from lu import *


def inv(a: Csr):
    l, u = decomp(a)
    for i in range(l.n):
        s, e = l.r[i : i + 2]
        for j in range(s, e):
            if l.c[j] != i:
                l.v[j] *= -1
    res = []
    for i in range(l.n - 1, -1, -1):
        res.append([(l[i, j] - sum(res[i - k][j] * u[k, i] for k in range(i + 1, l.n))) / u[i, i] for j in range(l.m)])
    return res[::-1]
