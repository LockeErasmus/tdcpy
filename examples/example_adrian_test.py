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
import tdspy.plot

from tdspy.stability.characteristic_roots import rightmost_root

# Set up logging
import logging
logger = logging.getLogger("tdspy")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
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
A = np.stack([A0, A1], axis=2)
hA = np.array([0,1.])
r = -3.0
ddae = tdspy.DDAE(A=A, hA=hA)
rdde = tdspy.RDDE(A=A, hA=hA)
cr, info = tdspy.roots(rdde, r=r)
cr_max = np.max(np.real(cr))

sa = tdspy.sa(ddae,r=-0.9)

if False:
    for method in ["hybr", "lm", "broyden1", "broyden2", "anderson", "linearmixing",
                "diagbroyden", "excitingmixing", "krylov", "df-sane"]:
        try:
            sa= tds.sa(ddae, scipy_root_method=method)
        except:
            print(f"{method=} FAILED")

print(f"strong spectral abscissa of associated DIFF {sa=}")
print(f"roots are cr {cr_max=}")

