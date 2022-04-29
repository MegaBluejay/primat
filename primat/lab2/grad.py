from functools import partial
from itertools import count

import numpy as np
from scipy.optimize import minimize_scalar
from toolz import comp


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
)
