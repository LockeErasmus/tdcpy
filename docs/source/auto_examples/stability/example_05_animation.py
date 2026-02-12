r"""
Discretization Animation
========================

In this example, we create an animation of the right-most part of the spectrum
obtained via method proposed in :cite:`jarlebring2010krylov`. The considered 
system is the following retarded delay differential equation (RDDE) from
:cite:`verheyden2008efficient` Section 6.1:

.. math::

    \dot{x}(t) =
        \begin{bmatrix}
            -1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & -10 & -4 \\
            0 & 0 & 4 & -10
        \end{bmatrix} x(t)
        + 
        \begin{bmatrix}
            3 & 3 & 3 & 3\\
            0 & -1.5 & 0 & 0 \\
            0 & 0 & 3 & -5 \\
            0 & 5 & 5 & 5
        \end{bmatrix} x(t-1).
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
# %%
# We use the build-in function for creating the discretization animation. 
# If further customization would be needed, we encourage to read the source
# code of `tdspy.plot.discretization_animation(.)` and adapt it.
# %%
ani = tdspy.plot.discretization_animation(
    rdde,
    s0=0,
    discretization=range(10, 100, 5),
    discretization_ideal=120,
    xlim=(-5.25, 1.5),
    ylim=(-200, 200),
    interval=200,
)
ani.save("discretization.gif", writer="pillow")
