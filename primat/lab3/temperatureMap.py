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
for k in range(0,4):
    for n in range(0,100,10):
        A, F = qq(n, k)
        time1 = time()
        x, count = z.zeydel(A,np.array(F),1)
        time2 = time()
        timeZeid.append(time2 - time1)
        print(x)
        time2 = time()
        l, u = lu.decomp(A.transpose())
        x = lu.solve(l, u, np.array(F))
        time3 = time()
        print(x)
        ns.append(n)
        ks.append(k)
        timeLU.append(time3-time2)

plt.scatter(ks,ns,c=timeLU,cmap='Greens', label='LU')
plt.colorbar()
plt.legend()
plt.xlabel('k')
plt.ylabel('n')
plt.show()
plt.scatter(ks,ns,c=timeZeid,cmap='Reds', label='Zeydel')
plt.colorbar()
plt.legend()
plt.xlabel('k')
plt.ylabel('n')
plt.show()
