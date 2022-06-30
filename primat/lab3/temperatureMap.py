import numpy as np
import primat.lab3.lu as lu
from gen import *
import matplotlib.pyplot as plt
import primat.lab3.csr as csr
from time import time
import zeydel as z
timeZeid = []
timeLU = []
ns = []
ks =[]
dxs = []
for n in range(10,100,10):
    A, F = gilbert(n)
    time1 = time()
    x, count = z.zeydel(A,np.array(F),0.01)

    time2 = time()
    dx = np.mean([abs(x[i] - i - 1) for i in range(len(x))])
    timeZeid.append(time2 - time1)
    time2 = time()
    l, u = lu.decomp(A.transpose())
    x = lu.solve(l, u, np.array(F))
    time3 = time()
    ns.append(n)
    dxs.append(dx)
    timeLU.append(time3-time2)

plt.plot(ns,timeLU)
plt.show()
plt.plot(ns,timeZeid)
plt.show()
plt.plot(ns,dxs)
plt.show()

