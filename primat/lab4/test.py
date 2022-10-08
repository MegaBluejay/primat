import numpy as np

from .jacobi import *
from ..lab3.gen import gilbert, qq


def test(a, eps, maxit=None):
    res = jacobi(Csr(a), eps, maxit)
    if isinstance(a, Csr):
        a = np.array(a.to_mat())
    vecs = np.array(res.vecs.to_mat())
    with_mat, with_vals = a @ vecs, res.vals * vecs
    if maxit is None:
        assert np.allclose(with_mat, with_vals)
    return np.sum((with_mat - with_vals) ** 2), res.its


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
    test(a, eps)


def gilbert_test(n, eps=None, maxit=None):
    return test(gilbert(n)[0], eps, maxit)


def qq_test(n, k, eps=None, maxit=None):
    a = np.array(qq(n, k)[0].to_mat())
    a += a.T
    return test(a, eps, maxit)
