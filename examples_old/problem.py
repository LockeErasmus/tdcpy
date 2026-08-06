
import tdcpy
import numpy as np
import matplotlib.pyplot as plt

tdcpy.init_logger(level="DEBUG")

epsilon = 2
assert epsilon > 0 and epsilon < np.pi, "epsilon has to be positive and smaller than pi"
sinc_epsilon = np.sin(epsilon)/epsilon

E = np.array([
    [1., 0],
    [0,  0]
])

A0 = np.array([
    [0., 0],
    [1./(2*epsilon), sinc_epsilon]
])
A1 = np.array([
    [0., 1],
    [0, 0]
])
A2 = np.array([
    [0., -1],
    [0, 0],
])
hA = np.array([0, np.pi-epsilon, np.pi+epsilon])

from tdcpy.stability.characteristic_roots import roots_ddae, RootsInfo

roots, info = roots_ddae(E, np.stack([A0, A1, A2], axis=2), hA, r=[-10, 10, -50, 50])

import tdcpy.plot

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
tdcpy.plot.eigen_plot(roots, ax=ax1)

plt.show()
