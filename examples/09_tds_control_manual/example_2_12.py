# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

r"""
Example 2.12 - Stability analysis of a Smith predictor with delay mismatch
==========================================================================

We will follow example 2.12 from :cite:`appeltans2023analysis` Section 2.6.2.
In there, for given plant and controller

.. math::

    H(s) = \frac{1}{s + 1}e^{-s\tau} , C(s) = \frac{s}{2} + 2,

the classical smith predictor is constructed. Assuming the delay mismatch
:math:`\delta` we arrive at the quasipolynomial

.. math::

    D_{cl}(s) = \frac{3}{2} s + 3 + (\frac{s}{2} + 2) e^{-s\tau} + 
        (\frac{-s}{2} - 2) e^{-s(\tau + \delta)},

which we will analyse for stability in :math:`(\tau, \delta)`-parameter
space.  
"""

import numpy as np
import tdspy
import tdspy.plot
from tdspy.common.quasipoly import qp_to_ndde
from tdspy.common.quasipoly import compress_qp, qp_to_ndde


tau, delta = 1., 0.5

coeffs = np.array([[3.,1.5],[2.0,0.5],[-2.0,-0.5]])
delays = np.array([0.,tau,tau+delta])

A, hA, H, hH = qp_to_ndde(coeffs,delays,ascending=True)

ndde = tdspy.NDDE(A=A,hA=hA,H=H,hH=hH)

tau_grid = np.linspace(0, 8, 201)
delta_grid = np.linspace(-8, 10, 451)
Z = np.zeros((len(delta_grid), len(tau_grid)))

for i2 in range(0,len(tau_grid)-1):
    tau = tau_grid[i2]
    for i1 in range(0,len(delta_grid)):
        delta = delta_grid[i1]
        if np.abs(tau+delta)<1e-8:
            # case: tau+delta = 0
            if np.abs(tau)<1e-8:
                tau = 1e-8
            hH[0],hH[1] = tau, 1e-8
            hA[1],hA[2] = tau, 1e-8
            ndde = tds.NDDE(H=H,hH=hH,A=A,hA=hA)
            Z[i1,i2] = tds.strong_spectral_abscissa(ndde)
        elif tau + delta < 0:
            # case: "real" delay cannot be negative
            Z[i1,i2] = -np.inf
        else:
            hH[0], hH[1] = tau, tau+delta
            hA[1], hA[2] = tau, tau+delta
            ndde = tds.NDDE(H=H,hH=hH,A=A,hA=hA)
            Z[i1,i2] = tds.strong_spectral_abscissa(ndde)

# X, Y = np.meshgrid(tau_grid,delta_grid)
# plt.contour(X,Y,Z,[0,0])
# plt.plot(plt.xlim,[0,0],'k-.')
# plt.show()