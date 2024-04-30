"""
Example 2.1 from the TDS-CONTROL manual
We will analyze the exponential stability of the following RDDE from [1,
Section 6.1]: 
x'(t) = A0 x(t) + A1 x(t-1)
with

     [-1 0   0   0]          [3   3  3  3]
A0 = [ 0 1   0   0] and A1 = [0 -1.5 0  0].
     [ 0 0 -10  -4]          [0   0  3 -5]
     [ 0 0   4 -10]          [0   5  5  5]

[1] Verheyden K., Luzyanina T., and Roose D. (2008). Efficient
    computation of characteristic roots of delay differential equations
    using LMS methods. Journal of Computational and Applied Mathematics,
    214(1), pp. 209–226.    
"""
import numpy as np
import tdspy
import tdspy.ddae
import tdspy.roots

# Set up logging
import logging
logger = logging.getLogger("tdspy")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# Create DDAE representation
A0 = np.array([[-1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, -10, -4],
               [0, 0, 4, -10]])
A1 = np.array([[3, 3, 3, 3],
               [0, -1.5, 0, 0],
               [0, 0, 3, -5],
               [0, 5, 5, 5]])


rdde = tdspy.ddae.DDAE(A=[A0, A1], hA=[0, 1.])
print(rdde.n)
cr = tdspy.roots.roots(rdde, r=-1.5)

print(cr)