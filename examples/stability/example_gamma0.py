"""

"""

import numpy as np

import tdspy
import tdspy.plot
import matplotlib.pyplot as plt

from tdspy.stability.spectral_abscissa import spectral_abscissa_diff    
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


diff = tdspy.DDAE(A=[D0, D1, D2, D3], hA=hD, E=np.zeros((4, 4)))

tdspy.init_logger(level="DEBUG")
# cr, info = tdspy.roots(diff, r=[-10, 10, -100, 100], discretization=200)
cr, info = tdspy.roots(diff, r=[-2, 0, -50, 50], discretization=200)

tdspy.plot.eigen_plot(info.newton_inital_guesses, title="Roots of difference equation", xlabel="Real part", ylabel="Imaginary part")
plt.show()





# from scipy import linalg

# print(linalg.inv(D0))

# # gamma0, info = tdspy.gamma(diff, r=0)
# # print(gamma0, info)

# P = -linalg.inv(D0)

# diff2 = tdspy.DDAE(A=[D0@P, D1@P, D2@P, D3@P], hA=hD, E=np.zeros((4, 4)))

# gamma0, info = tdspy.gamma(diff2, r=0)
# print(gamma0, info)
