def gilbert(n):
    return [[1/(i + j + 1) for j in range(n)] for i in range(n)]
