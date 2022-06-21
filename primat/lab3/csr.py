from bisect import bisect_left


class Csr:
    def __init__(self, m):
        if isinstance(m, Csr):
            self.v, self.c, self.r, self.m = m.v, m.c, m.r, m.m
            return
        if isinstance(m, tuple):
            self.v, self.c, self.r, self.m = m
            return
        self.v, self.c, self.r, self.m = [], [], [], len(m[0])
        for i, row in enumerate(m):
            self.r.append(len(self.v))
            for j, x in enumerate(row):
                if not x:
                    continue
                self.v.append(x)
                self.c.append(j)
        self.r.append(len(self.v))

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
        if self.c[q] == j:
            return self.v[q]
        return 0

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.get_row(item)
        return self.get_item(*item)

    def to_mat(self):
        return [self[i] for i in range(self.n)]

    def __matmul__(self, other):
        pass

    def __str__(self):
        return str((self.v, self.c, self.r))

    def __repr__(self):
        return str(self)
