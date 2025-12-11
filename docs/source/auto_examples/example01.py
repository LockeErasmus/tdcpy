r"""
Example 01
==========

Example 2.1 from the TDS-CONTROL manual
We will analyze the exponential stability of the following RDDE from [1,
Section 6.1]
"""

import matplotlib.pyplot as plt
import numpy as np
import tdspy
import tdspy.plot

# Create RDDE matrix representation
A0 = np.array([[-1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, -10, -4],
               [0, 0, 4, -10]])
A1 = np.array([[3, 3, 3, 3],
               [0, -1.5, 0, 0],
               [0, 0, 3, -5],
               [0, 5, 5, 5]])

rdde = tdspy.RDDE(A=[A0, A1], hA=[0, 1])
cr, info = tdspy.roots(rdde, r=-2.5)

tdspy.plot.eigen_plot(cr)
plt.show()

