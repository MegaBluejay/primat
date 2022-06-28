import numpy as np
from lu import *
#test1
mat = Csr([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
l,u = decomp(mat)


print(np.array(l.to_mat()))
print(np.array(u.to_mat()).transpose())


#test2
test = np.random.rand(10**2, 10**2)
mat = Csr(test)
l,u = decomp(mat)
# print(np.allclose(u.mul_cool(l).to_mat(),test.tolist()))

#diagonal dominancy



