"""
Tutorial - Discretization Heuristic II
======================================

TODO - work in progress
"""

import numpy as np

# Create RDDE matrix representation
E = np.eye(4)
A0 = np.array([[-1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, -10, -4],
               [0, 0, 4, -10]])
A1 = np.array([[3, 3, 3, 3],
               [0, -1.5, 0, 0],
               [0, 0, 3, -5],
               [0, 5, 5, 5]])
A = np.stack([A0, A1], axis=2) # This stacking along 3rd dimension is done automatically when using high level interface, it is for leveraging vectorization
hA = np.array([0, 1.])


shift = 0

hmax = hA[-1]
hK = hA / hmax
K = A * np.exp(-shift * hmax * hA)
K[:,:,0] -= E * shift * hmax

# Note, in the case of neutral system, it is essential to check that half-plane
# defined by `r` indeed contains finitely many roots, i.e. :math:`\gamma(r) < 1`.