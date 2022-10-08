import numpy as np

from .jacobi import *


def test_jacobi(a):
    vals, vecs = jacobi(Csr(a), 1e-9)
    vecs = np.array(vecs.to_mat())
    return np.allclose(a @ vecs, vals * vecs)


def small_test():
    a = np.array(
        [
            [4, -30, 60, -35],
            [-30, 300, -675, 420],
            [60, -675, 1620, -1050],
            [-35, 420, -1050, 700],
        ],
        dtype=float,
    )
    return test_jacobi(a)
