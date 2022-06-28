import numpy as np
import random as rnd
from scipy.sparse import csr_matrix


def generate_matrix(n):
    rows = []
    columns = []
    data = []
    for i in range(n):
        for j in range(n):
            a = rnd.random()
            if a > 0.6 or i == j:
                rows.append(i)
                columns.append(j)
                num = (int(rnd.random() * 10)) + 1
                data.append(num)

    return csr_matrix((data, (rows, columns)))


