from functools import partial, cache
from itertools import count

import numpy as np
from scipy.optimize import minimize_scalar, bracket, OptimizeResult
from toolz import comp


_epsilon = np.sqrt(np.finfo(float).eps)


@cache
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def norm_sq(a):
    return np.dot(a, a)


def next_grad(x, df, h):
    return x - h * df(x)


def const_h(c):
    def get_h(**kwargs):
        return c

    return get_h


def break_h(h0, eps, lam):
    def get_h(*, f, df, x, **kwargs):
        h = h0
        while not (f(x) - f(next_grad(x, df, h)) >= eps * h * norm_sq(df(x))):
            h *= lam
        return h

    return get_h


def fib_method(func, brack=None, args=(), xtol=_epsilon, **unknown):
    tol = xtol
    f = lambda x: func(*(x,) + args)
    if brack is None:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func, args=args)
    elif len(brack) == 2:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(func, *brack, args=args)
    elif len(brack) == 3:
        xa, xb, xc = brack
        xa, xc = sorted([xa, xc])
        if not (xa < xb < xc):
            raise ValueError("Not a bracketing interval.")
        fa, fb, fc = map(f, [xa, xb, xc])
        if not (fb < fa and fb < fc):
            raise ValueError("Not a bracketing interval.")
        funcalls = 3
    else:
        raise ValueError("Not a bracketing interval.")

    xa, xc = sorted([xa, xc])
    n = next(i for i in count() if fibonacci(i) * tol > xc - xa)
    w = (xc - xa) * fibonacci(n) / fibonacci(n + 2)
    x1, x2 = xa + w, xc - w
    for i in range(n):
        if f(x1) < f(x2):
            x2, xc = x1, x2
            x1 = xa + xc - x2
        else:
            xa, x1 = x1, x2
            x2 = xa + xc - x1
    res = (xa + xc) / 2
    funcalls += n
    return OptimizeResult(
        fun=f(res),
        nfev=funcalls,
        x=res,
        nit=n,
        success=True,
        message="Success",
    )


def steepest_h(method):
    def get_h(*, f, df, x, **kwargs):
        f1d = comp(f, partial(next_grad, x, df))
        return minimize_scalar(f1d, method=method).x

    return get_h


def stop_x(e):
    def stop(*, x, px, **kwargs):
        return norm_sq(x - px) <= e**2

    return stop


def stop_f(e):
    def stop(*, x, px, f, **kwargs):
        return abs(f(x) - f(px)) <= e

    return stop


def grad(f, df, x, get_h, stop, maxi=None):
    yield x
    px = x
    for i in count():
        if i == maxi:
            return
        x = next_grad(x, df, get_h(f=f, df=df, x=x))
        if stop(x=x, px=px, f=f):
            return
        yield x
        px = x


__all__ = (
    "next_grad",
    "const_h",
    "break_h",
    "steepest_h",
    "stop_x",
    "stop_f",
    "grad",
    "norm_sq",
    "fib_method",
)
