from random import choices

def gilbert(n):
    return [[1/(i + j + 1) for j in range(n)] for i in range(n)]

def qq(n):
    x = list(range(-4, 1))
    res = []
    for i in range(n):
        q = list(choices(x, k=n-1))
        q.insert(i, -sum(q) + 10**-n)
        res.append(q)
    return res
