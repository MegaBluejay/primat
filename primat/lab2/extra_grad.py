import numpy as np
from scipy.optimize import minimize_scalar

from grad import norm_sq


def extra_grad(f, df, x, method, stop, maxi=None):
    yield x
    n = x.shape[0]
    i = 0
    while 1:
        s = df(x)
        for j in range(n):
            i += 1
            if i == maxi:
                return
            lam = minimize_scalar(lambda lam: f(x - lam * s), method=method).x
            px = x
            x = x - lam * s
            omega = norm_sq(df(x)) / norm_sq(df(px))
            # omega = max(0, np.dot(df(x), df(x) - df(px)) / norm_sq(df(px)))
            s = df(x) - omega * s
            yield x
            if stop(x=x, px=px, f=f, s=s):
                return


__all__ = ("extra_grad",)
