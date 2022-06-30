from bisect import bisect_left
from itertools import groupby


class Csr:
    def __init__(self, m):
        if isinstance(m, Csr):
            self.v, self.c, self.r, self.m = m.v.copy(), m.c.copy(), m.r.copy(), m.m
            return
        if isinstance(m, tuple):
            self.v, self.c, self.r, self.m = m
            return
        self.v, self.c, self.r, self.m = [], [], [0], len(m[0])
        for i, row in enumerate(m):
            for j, x in enumerate(row):
                if not x:
                    continue
                self.v.append(x)
                self.c.append(j)
            self.r.append(len(self.v))

    @classmethod
    def from_clist(cls, clist, n, m=None):
        m = m or n
        clist.sort()
        v, c, r = [], [], [0]
        j = 0
        for i in range(n):
            while j < len(clist) and clist[j][0] == i:
                v.append(clist[j][2])
                c.append(clist[j][1])
                j += 1
            r.append(len(v))
        return cls((v, c, r, m))

    @property
    def n(self):
        return len(self.r) - 1
    
    def get_row(self, i):
        start, end = self.r[i : i + 2]
        row = [0] * self.m
        for v, c in zip(self.v[start:end], self.c[start:end]):
            row[c] = v
        return row

    def get_item(self, i, j):
        start, end = self.r[i : i + 2]
        q = bisect_left(self.c, j, start, end)
        if q < len(self.c) and self.c[q] == j:
            return self.v[q]
        return 0

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get_row(item)
        return self.get_item(*item)

    def to_mat(self):
        return [self[i] for i in range(self.n)]

    def mul_cool(self, other):
        v, c, r = [], [], [0]
        for i in range(self.n):
            for j in range(self.m):
                ks, es = self.r[i : i + 2]
                ko, eo = other.r[j : j + 2]
                res = 0
                while ks < es and ko < eo:
                    if self.c[ks] == other.c[ko]:
                        res += self.v[ks] * other.v[ko]
                        ks += 1
                        ko += 1
                    elif self.c[ks] < other.c[ko]:
                        ks += 1
                    else:
                        ko += 1
                if res:
                    v.append(res)
                    c.append(j)
            r.append(len(v))
        return Csr((v, c, r, self.m))

    def transpose(self):
        qs = []
        for i in range(self.n):
            start, end = self.r[i : i + 2]
            for vv, j in zip(self.v[start:end], self.c[start:end]):
                qs.append((j, i, vv))
        return self.from_clist(qs, self.n)

    def __str__(self):
        return str((self.v, self.c, self.r))

    def __repr__(self):
        return str(self)
