"""
NDDE example from MATLAB manual page 16, equation (2.20)

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

# Create NDDE representation
A = np.stack([
    np.array([[0.25]]),
    np.array([[-1./3]]),
], axis=2)
hA = np.array([0, 1.])
H = np.stack([
    np.array([[-0.75]]),
    np.array([[0.5]]),
], axis=2)

# Three set of delays
hH = np.array([1., 2])
hH = np.array([1., 2.05])
hH = np.array([1., 2.005])

ndde = tds.NDDE(A=A, hA=hA, H=H, hH=hH)

ddae = ndde.to_ddae().compress()

r = [-0.9, 0.2, -500, 500]
cr, info = tdspy.roots(ddae, r=r, max_size_evp=1500)

import matplotlib.pyplot as plt
tdspy.plot.eigen_plot(cr)
plt.show()

