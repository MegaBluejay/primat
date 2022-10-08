import numpy as np

from .jacobi import *
from ..lab3.gen import gilbert


def test_dense_correct(a, res):
    if isinstance(a, Csr):
        a = np.array(a.to_mat())
    vecs = np.array(res.vecs.to_mat())
    return np.allclose(a @ vecs, res.vals * vecs)


def small_test(eps=1e-9):
    a = np.array(
        [
            [4, -30, 60, -35],
            [-30, 300, -675, 420],
            [60, -675, 1620, -1050],
            [-35, 420, -1050, 700],
        ],
        dtype=float,
    )
    return test_dense_correct(a, jacobi(Csr(a), eps=eps))


def gilbert_test_correct(n, eps=1e-9):
    a = gilbert(n)[0]
    res = jacobi(a, eps=eps)
    return test_dense_correct(a, res)
