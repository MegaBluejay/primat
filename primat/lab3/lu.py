from csr import *


def gen_l(a: Csr, n):
    v, c, r = [], [], []
    for i in range(a.n):
        r.append(len(v))
        if i > n:
            v.append(-a[i, n] / a[n, n])
            c.append(n)
        v.append(1)
        c.append(i)
    r.append(len(v))
    return Csr((v, c, r, a.m))
