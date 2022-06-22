import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from grad import *
from extra_grad import *


def f1(x):
    # f1 = 10x^2+y^2
    return np.dot(x ** 2, [10, 1])


def df1(x):
    # grad1 = 20x+y
    return [20, 2] * x


def f2(x):
    # f2 = 10000x^2+300y^2
    return np.dot(x ** 2, [10000, 300])


def df2(x):
    # grad2 = 2000x+2y
    return [20000, 600] * x

def f3(x):
    #f3 = y^2/2
    return np.dot(x ** 2, [0.5, 0])
def df3(x):
    #df3 = y
    return [1, 0] * x

def rosenbrock(q):
    x, y = q
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def drosenbrock(q):
    x, y = q
    return np.array([-400 * x * (-(x ** 2) + y) + 2 * x - 2, -200 * (x ** 2) + 200 * y])


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
functext = ["10x^2+y^2", "10000x^2+300y^2"]
table = PrettyTable(["name", *functext])
funcs = [(f1,df1), (f2,df2)]
hs = [steepest_h("brent"), break_h(0.7, 0.1, 0.1), steepest_h("golden"),const_h(0.1)]
for h in hs:
    res1 = np.array(
    list(grad(f=f1, df=df1, x=np.array([100, 100]), get_h=h, stop=stop_f(1e-5))))
    res2 = np.array(
    list(grad(f=f2, df=df2, x=np.array([100, 100]), get_h=h, stop=stop_f(1e-5))))
    table.add_row(["break 0.1 0.1 0.95",res1.shape[0],res2.shape[0]]);
print(table)

