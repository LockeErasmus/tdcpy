"""
RDDE example for matrices with four delays

"""

import numpy as np
import tdspy as tds
import tdspy.plot
import cProfile

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
A2 = np.random.randint(-10,10, size=(4, 4))
A3 = np.random.randint(-10,10, size=(4, 4))
A = np.stack([A0, A1, A2, A3], axis=2)
hA = np.array([0,1.,2.,3.])

# Three set of delays)

ddae = tds.DDAE(A=A, hA=hA)


r = [-0.9, 0.2, -500, 500]
cr, info = tdspy.roots(ddae, r=r, max_size_evp=600)

import matplotlib.pyplot as plt
tdspy.plot.eigen_plot(cr)
plt.show()

