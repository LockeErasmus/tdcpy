# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adrian Saldanha

"""
Gamma0 computation for delay-difference equations
==================================================

"""

import numpy as np

import tdcpy
import tdcpy.plot
import matplotlib.pyplot as plt

from tdcpy.stability.spectral_abscissa import spectral_abscissa_diff    
D0 = np.array([[ 0., -1.,  0.,  0.],
          [-1.,  0.,  0.,  0.],
          [ 0.,  0.,  0., -1.],
          [ 0.,  0., -1.,  0.]])
D1 = np.array([[0., 0., 0., 0.],
       [0., 3., 0., 0.],
       [0., 4., 0., 0.],
       [0., 1., 0., 0.]])
D2 = np.array([[ 0. ,  0. ,  0. ,  0. ],
       [ 0. ,  0.4,  0. ,  0. ],
       [ 0. , -0.4,  0. ,  0. ],
       [ 0. , -0.4,  0. ,  0. ]])
D3 = np.array([[0.0365, 0.    , 0.0416, 0.048 ],
       [0.    , 0.    , 0.    , 0.    ],
       [0.    , 0.    , 0.    , 0.    ],
       [0.    , 0.    , 0.    , 0.    ]])

hD = np.array([0. , 2.5, 5. , 0. ])
hD = np.array([0. , 2.5, 5.05 , 0. ])


diff = tdcpy.DDAE(A=[D0, D1, D2, D3], hA=hD, E=np.zeros((4, 4)))

tdcpy.init_logger(level="DEBUG")
# cr, info = tdcpy.roots(diff, r=[-10, 10, -100, 100], discretization=200)
cr, info = tdcpy.roots(diff, r=[-2, 0, -50, 50], discretization=200)

tdcpy.plot.eigen_plot(info.newton_inital_guesses, title="Roots of difference equation", xlabel="Real part", ylabel="Imaginary part")
plt.show()





# from scipy import linalg

# print(linalg.inv(D0))

# # gamma0, info = tdcpy.gamma(diff, r=0)
# # print(gamma0, info)

# P = -linalg.inv(D0)

# diff2 = tdcpy.DDAE(A=[D0@P, D1@P, D2@P, D3@P], hA=hD, E=np.zeros((4, 4)))

# gamma0, info = tdcpy.gamma(diff2, r=0)
# print(gamma0, info)
