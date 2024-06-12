"""
Example 2.2 from TDS-CONTROL manal
We will analyze the following rdde from [1], section 6.1:
x'(t) = A0 x(t) + A1 x(t-\tau_2) + H1 \dot{x}(t-\tau_1)
with 

               
A0 = [-0.6  -0.45 ] and A1 =   [-0.15  0.075], \tau_2 = 2.
     [ 0.1  -1.2  ]            [0.225  -0.75]

H1 = [3   -3/2 ], \tau_1 = 1
     [5/2    1 ]

[1] Verheyden K., Luzyanina T., and Roose D. (2008). Efficient
    computation of characteristic roots of delay differential equations
    using LMS methods. Journal of Computational and Applied Mathematics,
    214(1), pp. 209–226.
"""

import numpy as np
import tdspy as tds
import tdspy.plot

# Set up logging
import logging
logger = logging.getLogger("tdspy")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Create DDAE representation
A0 = np.array([[-0.6, -0.5],
               [0.1,-1.2]])

A1 = np.array([[-0.15,0.075],
               [0.225,-0.75]])

A = np.stack([A0,A1],axis = 2)
hA = np.array([0,2.])

H1 = np.array([[3, -1.5],
               [2.5,-1]])
H = np.stack([H1],axis=2)
hH = np.array([1])

r = -2.5
ndde = tdspy.NDDE(A=A, hA=hA,H=H, hH=hH)
cr, cr0 = tdspy.roots(ndde, r=-2.5)

diff = ndde.get_delay_difference_equation()

print(diff)
print(f"E={diff.E=}, A={diff.A}, hA={diff.hA=}")

cd = tds.cd(ndde)
print(f"strong spectral abscissa of associated DIFF {cd=}")

# sa = tds.strong_sa(ndde)
# print(f"spectral abscissa of the ndde {sa=}")

# import matplotlib.pyplot as plt

# tdspy.plot.eigen_plot(cr)
# plt.show()