from math import sin
from primat.primat import *

f = lambda x: sin(x) - log(x**2) - 1


def test(n):
    res = {}
    for minimizer in [DihotMinimizer, GoldenMinimizer, FibMinimizer, ParabolaMinimizer, BrentMinimizer]:
        try:
            res[minimizer.__name__] = minimizer(f, 8, 13.5, 10 ** (-5), max_calls=n).minimize()
        except Exception:
            res[minimizer.__name__] = "failed"
    return res


table = {i: test(i) for i in range(4, 50)}
for i in range(4, 50):
    print(i)
    for name, res in table[i].items():
        print(" ", name, res)
