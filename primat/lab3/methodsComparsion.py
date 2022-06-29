import numpy as np
import primat.lab3.lu as lu
from gen import *
import primat.lab3.csr as csr
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

for k in range(0,1):
    r = []
    dr = []
    A, F = almost_diag(1_000_000)
    x, count = z.zeydel(A,np.array(F),1)
    dx = [x[i]-(i+1) for i in range(len(x))]
    dr.append(np.mean(dx))
    d.append(dr)
print(d)
print(count)

