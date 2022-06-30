import numpy as np
import primat.lab3.lu as lu
from gen import *
import matplotlib.pyplot as plt
import primat.lab3.csr as csr
from time import time
import zeydel as z
table = []
d = []
# for k in range(-1,0):
#     r = []
#     dr = []
#     A, F = qq(700, k)
#     x, count = z.zeydel(A,np.array(F),1)
#     dx = [x[i]-(i+1) for i in range(len(x))]
#     dr.append(np.mean(dx))
#     d.append(dr)
# print(d)
# print(count)
# for n in range(5,30,5):
#     A, F = gilbert(n)
#     x,count = z.zeydel(A,np.array(F),0.01)
#     dx = [abs(x[i] - (i + 1)) for i in range(len(x))]
#     print('steps='+str(count))
#     print('dx='+str(np.mean(dx)))

#almost diagonal
times = []
ns = []
for n in range(0,200,10):
    print(str(round(n/200*100,2))+'%')
    r = []
    dr = []
    A, F = almost_diag(n)
    t1 = time()
    A, F = almost_diag(n)
    l, u = lu.decomp(A.transpose())
    x = lu.solve(l, u, np.array(F))
    t2 = time()
    ns.append(n)
    times.append(t2-t1)
plt.plot(ns,times)
plt.xlabel('n')
plt.ylabel('time')
plt.show()
A, F = almost_diag(100)
l,u = lu.decomp(A.transpose())
x = lu.solve(l,u,np.array(F))
print(x)
dx = [abs(x[i]-(i+1)) for i in range(len(x))]
print(np.mean(dx))
