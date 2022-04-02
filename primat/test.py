from primat import *
from org_table import table
import pandas as pd

f = lambda x: sin(x) - log(x**2) - 1
minimizers = [DihotMinimizer, GoldenMinimizer, FibMinimizer, ParabolaMinimizer, BrentMinimizer]


def test(n, minimizer):
    return minimizer(f, 8, 13.5, 10 ** (-5), max_calls=n).minimize()


ns = range(4, 50)
index = pd.MultiIndex.from_product([ns, [m.__name__ for m in minimizers]], names=["n", "Minimizer"])
df = pd.DataFrame([test(n, m) for n in ns for m in minimizers], index=index)
with open("res.org", "w") as file:
    print(table(df, fmt=".8f"), file=file)
