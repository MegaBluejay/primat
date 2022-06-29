import numpy as np
import primat.lab3.lu as lu
import primat.lab3.csr as csr
#test1
mat = csr.Csr([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
l,u = lu.decomp(mat)

print(np.array(mat.to_mat()))
print(np.array(l.to_mat()))
print(np.array(u.to_mat()).transpose())


#test2
test = np.random.rand(10**1, 10**1)
mat1 = csr.Csr(test)
l,u = lu.decomp(mat1)
print(np.allclose(u.mul_cool(l).to_mat(),test.tolist()))

#inverse

print(np.array(lu.inv(mat).to_mat()))
print(np.matmul(np.array(lu.inv(mat).to_mat()),np.array(mat.to_mat())))





