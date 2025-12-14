r"""
Discretization Animation
========================

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

# ani = tdspy.plot.discretization_animation(
#     rdde,
#     s0=0,
#     discretization=range(10,85),
#     discretization_ideal=100,
#     xlim=(-3.25,1.5),
#     ylim=(-200, 200),
# )
# plt.show()
