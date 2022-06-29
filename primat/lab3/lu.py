from primat.lab3.csr import *


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
    u = u.transpose()
    n = l.n
    y = []
    for i in range(n):
        start, end = l.r[i : i + 2]
        yv = b[i]
        for v, j in zip(l.v[start:end], l.c[start:end]):
            if j >= i:
                break
            yv -= y[j] * v
        y.append(yv / l[i, i])

    x = []
    for i in range(n):
        xv = y[n - i - 1]
        start, end = u.r[n - i - 1 : n - i + 1]
        for v, j in zip(u.v[start:end][::-1], u.c[start:end][::-1]):
            if n - j - 1 >= len(x):
                break
            xv -= x[n - j - 1] * v
        x.append(xv / u[n - i - 1, n - i - 1])
        # x.append((y[n - i - 1] - sum(x[j] * u[n - j - 1, n - i - 1] for j in range(i))) / u[n - i - 1, n - i - 1])
    return x[::-1]


def inv(a):
    n = a.n
    l, u = decomp(a)
    v, c, r = [], [], [0]
    for j in range(n):
        x = [0] * n
        x[j] = 1
        col = solve(l, u, x)
        for i, vv in enumerate(col):
            if vv:
                v.append(vv)
                c.append(i)
        r.append(len(v))
    return Csr((v, c, r, n))
