import numpy as np
from numpy.lib.stride_tricks import as_strided 

a = np.arange(4*4).reshape(4,4)
u = a.itemsize
b = as_strided(a, shape=(3,3,2,2), strides=(4*u, u, 4*u, u)) 
c = np.ones((2,2))
print(a)
print(np.tensordot(b, c, axes=[[2,3],[0,1]]))