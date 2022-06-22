from bisect import bisect_left

from csr import *


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
    v, c, r = [], [], []
    for i in range(a.n):
        r.append(len(v))
        for j in range(i):
            k = bisect_left(rs[j], i)
            if rs[j][k] == i:
                v.append(vs[j][k])
                c.append(j)
        v.append(1)
        c.append(i)
    r.append(len(v))
    return Csr((v, c, r, a.m)), a
