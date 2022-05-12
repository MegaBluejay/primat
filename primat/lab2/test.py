import numpy as np
import matplotlib.pyplot as plt

from grad import *
from extra_grad import *


def f1(x):
    return np.dot(x**2, [10, 1])


def df1(x):
    return [20, 2] * x


res = np.array(list(grad(f=f1, df=df1, x=np.array([10, 10]), get_h=steepest_h("golden"), stop=stop_f(1e-5))))
print(res.shape[0], f"{np.sqrt(norm_sq(res[-1])):e}")

x = np.mgrid[-15:15:100j, -15:15:100j]
z = f1(np.moveaxis(x, 0, -1))
plt.contour(*x, z)
plt.plot(*res.swapaxes(0, 1))
plt.show()
