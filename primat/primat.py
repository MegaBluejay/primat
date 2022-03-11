from itertools import count
from functools import cache
from math import *
from collections import namedtuple


class CountingFunc:
    def __init__(self, f):
        self.f = f
        self.n = 0

    def __call__(self, *args, **kwargs):
        self.n += 1
        return self.f(*args, **kwargs)


MinResult = namedtuple("MinResult", ["x", "calls", "steps"])


class Minimizer:
    def __init__(self, f, a, b, eps, max_calls=0):
        self.cf = CountingFunc(f)
        self.f = cache(self.cf)
        self.a = a
        self.b = b
        self.eps = eps
        self.max_calls = max_calls or inf

    def next_step(self):
        raise NotImplementedError

    def minimize(self):
        steps = 0
        while self.cf.n < self.max_calls and self.b - self.a >= self.eps * 2:
            steps += 1
            self.next_step()
        print(self.__class__.__name__, self.b - self.a)
        return MinResult(x=(self.a + self.b) / 2, calls=self.cf.n, steps=steps)


class DihotMinimizer(Minimizer):
    def __init__(self, *args, delta_mod=0.3, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = self.eps * delta_mod

    def next_step(self):
        mid = (self.a + self.b) / 2
        x1, x2 = mid - self.delta, mid + self.delta
        if self.f(x1) < self.f(x2):
            self.b = x2
        else:
            self.a = x1


class GoldenMinimizer(Minimizer):
    q = (3 - sqrt(5)) / 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        w = self.q * (self.b - self.a)
        self.x1 = self.a + w
        self.x2 = self.b - w

    def next_step(self):
        if self.f(self.x1) < self.f(self.x2):
            self.b, self.x2 = self.x2, self.x1
            self.x1 = self.a + self.b - self.x2
        else:
            self.a, self.x1 = self.x1, self.x2
            self.x2 = self.a + self.b - self.x1


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
