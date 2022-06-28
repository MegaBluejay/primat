from .csr import *


def gen_l(a: Csr, n):
    v, c, r = [], [], []
    resv, resr = [], []
    for i in range(a.n):
        r.append(len(v))
        if i > n:
            q = -a[n, i] / a[n, n]
            if q:
                resv.append(q)
                resr.append(i)
                v.append(resv[-1])
                c.append(n)
        v.append(1)
        c.append(i)
    r.append(len(v))
    return Csr((v, c, r, a.m)), resv, resr


def decomp(a: Csr):
    vs, rs = [], []
    for i in range(a.n):
        li, resv, resr = gen_l(a, i)
        vs.append(resv)
        rs.append(resr)
        a = a.mul_cool(li)
    v, c, r = [], [], [0]
    for i in range(a.n):
        for j in range(i):
            k = bisect_left(rs[j], i)
            if rs[j][k] == i:
                v.append(-vs[j][k])
                c.append(j)
        v.append(1)
        c.append(i)
        r.append(len(v))
    return Csr((v, c, r, a.m)), a


def solve(l, u, b):
    n = l.n
    y = []
    for i in range(n):
        y.append((b[i] - sum(y[j] * l[i, j] for j in range(i))) / l[i, i])
    x = []
    for i in range(n):
        x.append((y[n - i - 1] - sum(x[j] * u[n - j - 1, n - i - 1] for j in range(i))) / u[n - i - 1, n - i - 1])
    return x[::-1]
