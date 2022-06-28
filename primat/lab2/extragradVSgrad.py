import numpy as np
from grad import *
from extra_grad import *
import matplotlib.pyplot as plt
sz=50
fvs = []
tests1 = 50
coef1 = 1
coef2 = 0.001
tests2 = 50
tests = 300
while(len(fvs)!=tests):
    fv = np.random.rand(1, sz) * coef1
    k = np.linalg.cond(np.diag(fv.tolist()[0]), p="fro")
    if(k<150):
        print('gen!')
        fvs.append(fv)
# for i in range(tests1):
#     fvs.append(np.random.rand(1, sz) * coef1)
# for i in range(tests2):
#     fvs.append(np.random.rand(1, sz) * coef2)
hname,hfunc = ('golden','golden')
ns=[]
ks=[]
stepsgrad = []
stepsExtragrad = []
for i in range(tests):
    print(str(round(i*100/tests, 2))+'%')
    fv = fvs[i]
    dfv = fv*2
    fv = fv.tolist()[0]
    dfv = dfv.tolist()[0]
    k = np.linalg.cond(np.diag(fv), p="fro")
    def f(x):
        return np.dot(x ** 2, fv)
    def df(x):
        return dfv * x

    res1 = np.array(
                     list(grad(f=f, df=df, x=np.array([10 for t in range(sz)]), get_h=steepest_h('golden'), stop=stop_f(1e-5)))
                )
    res2 = np.array(
                     list(extra_grad(f=f, df=df, x=np.array([10 for t in range(sz)]), method='golden', stop=stop_f(1e-5)))
                )
    ks.append(k)
    stepsgrad.append(res1.shape[0])
    stepsExtragrad.append(res2.shape[0])
plt.scatter(ks,stepsgrad,c='b',label='grad')
plt.scatter(ks,stepsExtragrad,c='r',label='conjugate grad')
plt.xlabel('condition number')
plt.ylabel('steps')
plt.legend(loc='upper left')
plt.show()