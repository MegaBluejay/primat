import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from grad import *
from extra_grad import *


def f1(x):
    # f1 = 10x^2+y^2
    return np.dot(x**2, [10, 1])


def df1(x):
    # grad1 = 20x+y
    return [20, 2] * x


def f2(x):
    # f2 = 10000x^2+10000y^2
    return np.dot(x**2, [10000, 10000])


def df2(x):
    # grad2 = 2000x+2y
    return [20000, 20000] * x


def f3(x):
    # f3 = y^2/2
    return np.dot(x**2, [100000, 0.00001])


def df3(x):
    # df3 = y
    return [200000, 0.00002] * x


def rosenbrock(q):
    x, y = q
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def drosenbrock(q):
    x, y = q
    return np.array([-400 * x * (-(x**2) + y) + 2 * x - 2, -200 * (x**2) + 200 * y])


# res = np.array(
#     list(grad(f=rosenbrock, df=drosenbrock, x=np.array([0, 0]), get_h=steepest_h("brent"), stop=stop_f(1e-5))))
# print(res.shape[0], f"{np.sqrt(norm_sq(res[-1])):e}")
# print(res[-1])
# x = np.mgrid[-2:2:100j, -2:2:100j]
# #z = rosenbrock(np.moveaxis(x, 0, -1))
# z = np.array(list(map(rosenbrock, x.reshape((-1, 2))))).reshape((100, 100))
# plt.contour(*x, z)
# plt.plot(*res.swapaxes(0, 1))
# plt.show()
# step_size = [x / 1000 for x in range(1, 100)]
# step_count = [np.array(list(grad(f=f1, df=df1, x=np.array([10, 10]), get_h=const_h(step), stop=stop_f(1e-5)))).shape[0]
#               for step in step_size]
# plt.plot(step_size, step_count)
# plt.ylabel("к-во итераций")
# plt.xlabel("длина шага")
# plt.show()
functext = ["10x^2+y^2", "10000x^2+10000y^2", "100000x^2+0.00001y^2"]

funcs = [(f1, df1), (f2, df2), (f3, df3)]
start_points = [[10, 10], [1, 100], [1, 10000], [100, 1], [10000, 1], [1000, 10000]]
hs = [
    ("brent", steepest_h("brent")),
    ("break h0=0.7 eps=0.1 lambda=0.95", break_h(0.7, 0.1, 0.1)),
    ("break h0=0.5 eps=0.9 lambda=0.9", break_h(0.5, 0.9, 0.9)),
    ("golden", steepest_h("golden")),
    ("fibonacci", steepest_h(fib_method)),
]
for point in start_points:
    print("Начальная точка: ", point)
    table = PrettyTable(["название метода", *functext])
    for h in hs:
        hname, hfunc = h
        res1 = np.array(list(grad(f=f1, df=df1, x=np.array(point), get_h=hfunc, stop=stop_f(1e-5))))
        res2 = np.array(list(grad(f=f2, df=df2, x=np.array(point), get_h=hfunc, stop=stop_f(1e-5))))
        res3 = np.array(list(grad(f=f3, df=df3, x=np.array(point), get_h=hfunc, stop=stop_f(1e-5))))
        table.add_row([hname, res1.shape[0], res2.shape[0], res3.shape[0]])
    print(table)


# condition numbers
funcs = [
    [[10, 1], [100, 0.1], [10000, 0.0001]],
    [[1, 1, 1], [1, 100, 10000], [0.01, 1000, 100000]],
    [[3, 4, 5, 6, 7, 8], [0.1, 1, 10, 100, 1000, 10000], [0.01, 10, 100, 1000, 2000, 20000]],
]
pairs = []
for nfuncs in funcs:
    t = []
    for func in nfuncs:
        dfunc = 2 * np.array(func)
        t.append(tuple([func, dfunc.tolist()]))

    pairs.append(t)
for h in hs:
    hname, hfunc = h
    print("Method:" + hname)
    for nfuncs in pairs:
        sz = len(nfuncs[0][0])
        print("size: " + str(sz))
        fs = lambda i: lambda x: np.dot(x**2, nfuncs[i][0])
        dfs = lambda i: lambda x: nfuncs[i][1] * x
        cond = [np.linalg.cond(np.diag(a[0]), p="fro") for a in nfuncs]
        for i in range(len(nfuncs)):
            res = np.array(
                list(grad(f=fs(i), df=dfs(i), x=np.array([10 for t in range(sz)]), get_h=hfunc, stop=stop_f(1e-5)))
            )
            print("k=" + str(cond[i]) + " steps=" + str(res.shape[0]))
