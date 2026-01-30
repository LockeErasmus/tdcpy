"""
Discretization tutorial
=======================

In this tutorial, we will follow Example 5 from :cite:`wu2012reliably` to illustrate how
to obtain number of discretization points such as to reliably compute all roots of LTI time-delay
system in specifief right half-plane.

"""

import matplotlib.pyplot as plt
import numpy as np
import tdspy
import tdspy.plot

from tdspy.stability.discretization_heuristic import compute_n_rhp, incommensurate_gk
from tdspy.common.operations import shift_ddae, normalize_ddae

# %%
# Note that as this is tutorial, we will go deeper than high level interface and
# use internal functions.
# Note: we assume that matrices are non empty and delays are sorted unique with :math:`h_{A,0} = 0`, this preprocessing is done even if the :math:`A_0=0`.
# ------

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

n_grid = 20
r = -1.5

# Step 0: normalize by `hA[-1]` (maximal delay) and shift by `r` matrix representation, i.e.
# perform :math:`s \leftarrow \frac{s}{\tau_{max}} + r` on the DDAE representation
#
# .. math::
#     E s X(s) = ( A_0 + \sum_{i=1}^{mA} A_i e^{-h_{A,i} s} ) X(s)
# 
# In this example, we have only one delay :math:`\tau_{1} = 1`, so normalization by maximal delay is already satisfied.
# ------

hmax = hA[-1]
shift = -1.5

K = A * np.exp(-shift * hmax * hA)
K[:,:,0] -= E * shift * hmax


# Note, in the case of neutral system, it is essential to check that half-plane
# defined by `r` indeed contains finitely many roots, i.e. :math:`\gamma(r) < 1`.







# %%










