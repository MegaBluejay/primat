from itertools import count
from functools import cache
from math import *
from collections import namedtuple

from primat.base import CountingFunc


MinResult = namedtuple("MinResult", ["x", "dx", "calls", "steps"])


class Minimizer:
    def __init__(self, f, a, b, eps, max_calls=0):
        self.cf = CountingFunc(f)
        self.f = cache(self.cf)
        self.a = a
        self.b = b
        self.eps = eps
        self.max_calls = max_calls or inf

    def do_step(self):
        raise NotImplementedError

    def minimize(self):
        steps = 0
        while self.cf.n < self.max_calls and self.b - self.a >= self.eps * 2:
            steps += 1
            self.do_step()
        return MinResult(x=(self.a + self.b) / 2, dx=(self.b - self.a) / 2, calls=self.cf.n, steps=steps)


class DihotMinimizer(Minimizer):
    def __init__(self, *args, delta_mod=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = self.eps * delta_mod

    @staticmethod
    def next_step(f, delta, a, b):
        mid = (a + b) / 2
        x1, x2 = mid - delta, mid + delta
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
        return a, b

    def do_step(self):
        self.a, self.b = self.next_step(self.f, self.delta, self.a, self.b)


class GoldenMinimizer(Minimizer):
    q = (3 - sqrt(5)) / 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        w = self.q * (self.b - self.a)
        self.x1 = self.a + w
        self.x2 = self.b - w

    @staticmethod
    def next_step(f, a, b, x1, x2):
        if f(x1) < f(x2):
            b, x2 = x2, x1
            x1 = a + b - x2
        else:
            a, x1 = x1, x2
            x2 = a + b - x1
        return a, b, x1, x2

    def do_step(self):
        self.a, self.b, self.x1, self.x2 = self.next_step(self.f, self.a, self.b, self.x1, self.x2)


@cache
def fib(n):
    if n < 2:
        return 1
    return fib(n - 1) + fib(n - 2)


class FibMinimizer(GoldenMinimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinf(self.max_calls):
            n = next(k for k in count() if self.b - self.a < self.eps * fib(k + 2))
        else:
            n = self.max_calls
        w = (self.b - self.a) * fib(n) / fib(n + 1)
        self.x1 = self.b - w
        self.x2 = self.a + w


class ParabolaMinimizer(Minimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x2 = (self.a + self.b) / 2

    @staticmethod
    def next_step(f, a, x2, b):
        u = x2 - ((x2 - a) ** 2 * (f(x2) - f(b)) - (x2 - b) ** 2 * (f(x2) - f(a))) / (
            2 * ((x2 - a) * (f(x2) - f(b)) - (x2 - b) * (f(x2) - f(a)))
        )
        if f(x2) < f(u):
            if x2 < u:
                b = u
            else:
                a = u
        else:
            if x2 < u:
                a, x2 = x2, u
            else:
                x2, b = u, x2
        return a, x2, b

    def do_step(self):
        self.a, self.x2, self.b = self.next_step(self.f, self.a, self.x2, self.b)


class BrentMinimizer(Minimizer):
    k = (3 - sqrt(5)) / 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x = self.w = self.v = (self.a + self.b) / 2
        self.d = self.e = self.b - self.a

    @staticmethod
    def next_step(f, x, w, v, d, e, a, b, k, eps):
        g, e = e, d
        u = None
        if len({x, w, v}) == 3 and len(set(map(f, [x, w, v]))):
            x1, x2, x3 = sorted([x, w, v])
            u = x2 - ((x2 - x1) ** 2 * (f(x2) - f(x3)) - (x2 - x3) ** 2 * (f(x2) - f(x1))) / (
                2 * ((x2 - x1) * (f(x2) - f(x3)) - (x2 - x3) * (f(x2) - f(x1)))
            )
            if a + eps <= u <= b - eps and abs(u - x) < g / 2:
                d = abs(u - x)
            else:
                u = None
        if u is None:
            if x < (b - a) / 2:
                u = x + k * (b - x)
                d = b - x
            else:
                u = x - k * (x - a)
                d = x - a
        if abs(u - x) < eps:
            u = x + copysign(eps, u - x)
        if f(u) <= f(x):
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
        else:
            if u >= x:
                b = u
            else:
                a = u
            if f(u) <= f(w) or w == x:
                v, w = w, u
            elif f(u) <= f(v):
                v = u
        return a, b, x, w, v, d, e

    def do_step(self):
        self.a, self.b, self.x, self.w, self.v, self.d, self.e = self.next_step(
            self.f, self.x, self.w, self.v, self.d, self.e, self.a, self.b, self.k, self.eps
        )
