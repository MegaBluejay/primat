import numpy as np
import matplotlib.pyplot as plt



from functools import partial

from grad import *
from extra_grad import *
def f(x,fv):
    return np.dot(x ** 2, fv)


def df(x,dfv):
    return dfv * x
hs = [
    ("brent", steepest_h("brent")),
    ("break h0=0.7 eps=0.1 lambda=0.95", break_h(0.7, 0.1, 0.1)),
    ("break h0=0.5 eps=0.9 lambda=0.9", break_h(0.5, 0.9, 0.9)),
    ("golden", steepest_h("golden")),
    ("fibonacci", steepest_h(fib_method)),
]
fvs = []
coef1 = 1000
tests = 100
szs = 10
for h in hs:
    hname,hfunc = h
    ns=[]
    ks=[]
    steps = []

    for sz in range(2, szs):
        print(str(round(100*(sz-1)/(szs-1),2))+'%')
        fvs = []

        while (len(fvs) != tests):
            fv = np.random.rand(1, sz) * coef1
            k = np.linalg.cond(np.diag(fv.tolist()[0]), p="fro")
            if (k < 20):
                fvs.append(fv)
        for i in range(tests):
            fv = fvs[i]
            dfv = fv*2
            fv = fv.tolist()[0]
            dfv = dfv.tolist()[0]
            k = np.linalg.cond(np.diag(fv), p="fro")

            res = np.array(
            list(grad(f=partial(f, fv=fv), df=partial(df, dfv=dfv), x=np.array([10 for t in range(sz)]), get_h=hfunc, stop=stop_f(1e-5)))
            )
            ks.append(k)
            ns.append(sz)
            steps.append(res.shape[0])
    plt.scatter(ks,ns,c=steps,cmap='Greens', label='steps')
    plt.xlabel('cond')
    plt.ylabel('n')
    plt.legend()
    plt.title(hname)
    plt.colorbar()
    plt.show()