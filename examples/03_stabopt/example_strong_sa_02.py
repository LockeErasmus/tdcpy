r"""
Example - strong stabilization
==================================

We consider the system described by the following delay-differential equation from :cite:`michielsStability2007` and 
Example 3.2 of :cite:`appeltans2023analysis`
    

.. math::
    
    \begin{aligned}
        \dot{x}(t) &=
        \begin{bmatrix}
        -0.08 & -0.03 & 0.2 \\ 
        0.2  & -0.04 & -0.005 \\ 
        -0.06 &  0.2  & -0.07
        \end{bmatrix} x(t)
        +
        \begin{bmatrix}
        -0.1 \\
        -0.2 \\
        0.1
        \end{bmatrix} u(t-5), \\
        \\
        y(t) &=
        \begin{bmatrix}
        1. & 0. & 0. \\
        0. & 1. & 0. \\
        0. & 0. & 1. \\
        \end{bmatrix} x(t)
        +
        \begin{bmatrix}
        3. \\
        4. \\
        1.
        \end{bmatrix} u(t-2.5)
        +
        \begin{bmatrix}
        0.4 \\
        -0.4 \\
        -0.4
        \end{bmatrix} u(t-5).
    \end{aligned}

with 

.. math::

    u(t)   =   D_c y(t)

The goal is to compute the controller parameters Dc such that the closed-loop system is strongly stable.

We shall follow the following steps for designing our stabilizing controller:

1. Create the open-loop system as a `DDAE` object
2. Create the controller as a `DDAE` object
3. Form the closed-loop system as a `ClosedLoop` object
4. Optimize the controller parameters using `controller_bfgs` to minimize the spectral abscissa of the closed-loop system.

"""
import numpy as np
import matplotlib.pyplot as plt

import tdcpy
from tdcpy import DDAE
import tdcpy.plot

# %%
# **Step 1:** Create the open-loop `DDAE`
#

A0 = np.array([
    [   -0.08,  -0.03,  0.2     ],
    [   0.2,    -0.04,  -0.005  ],
    [   -0.06,  0.2,    -0.07    ],
])
Bu = np.array([
    [-0.1],
    [-0.2],
    [ 0.1],
])
C = np.array(np.eye(3)) # all states as outputs
D1 = np.array([
    [3.],
    [4],
    [1],
])
D2 = np.array([[0.4], [-0.4], [-0.4]])
hD = np.array([2.5,5.])

plant = tdcpy.DDAE(
    A=[A0], hA=[0],
    B=[Bu], hB=[5.],
    C=[C], hC=[0],
    D=[D1, D2], hD=[2.5, 5],
)

print(plant)

# %% 
# Let us examine the open-loop system, is the system stable?

from tdcpy import spectral_abscissa
sa = spectral_abscissa(plant, r=-1)

print(f"Spectral abscissa of open-loop: {sa}")


# %%
# Since the spectral abscissa is positive => the system is unstable. 
# Let us try to stabilize it by static output feedback.
#
# **Step 2:** Create the controller as a `DDAE` object and forming the closed-loop

from tdcpy.common.closed_loop import controller_reprezentation
from tdcpy import ClosedLoop

E, K, hK = controller_reprezentation(order=0, n_inputs=3, n_outputs=1, hD=np.array([0.]))

cl = ClosedLoop(plant, order=0, y_indices=[0,1,2], u_indices=[0], K0=np.zeros_like(K), hK=hK)

# %% 
# Let us first check if the closed-loop is retarded or neutral
cl_ddae = DDAE(E=cl.E,A=cl.A,hA=cl.hA)          # extracting the closed-loop ddae
print(f"Is the closed-loop system retarded? {cl_ddae.is_essentially_retarded}")


# %%
# Since the closed-loop is neutral, the controller will have to be designed for strong stability.


from tdcpy import strong_spectral_abscissa
ssa = strong_spectral_abscissa(cl_ddae, r=-1)

print(f"Strong spectral abscissa before optimization is: {ssa}")
region = [-0.5, 0.5, -0.5, 0.5]

cr_system, _ = tdcpy.roots(cl_ddae, r=region, discretization=100)
import matplotlib.pyplot as plt
import tdcpy.plot
fig, ax = plt.subplots()

tdcpy.plot.eigen_plot(cr_system, ax=ax)
ax.axvline(x=ssa, color="red", linestyle="--")
ax.text(ssa, 0, f"CD = {ssa:.4f}", color="red", fontsize=10,
        verticalalignment="bottom", horizontalalignment="right")
ax.title = ax.set_title("Open-loop roots and strong spectral abscissa")
ax.set_xlim(region[0], region[1])
ax.set_ylim(region[2], region[3])
ax.set_xlabel(r"$\Re (\lambda)$")
ax.set_ylabel(r"$\Im (\lambda)$")
plt.show()

# %%
# **Step 3:** Mimimizing the strong spectral abscissa of the closed-loop
# 
# Note that the strong spectral abscissa is non-smooth and non-convex function of the controller parameters.
# To this end, the high-level function `minimize_spectral_abscissa` provides an interface for minimizing the 
# strong spectral abscissa of the closed-loop system
# from function utilizes the `minimize` function from the `scipy.optimize` module. The default
# is the "L-BFGS-B" method.

from tdcpy.stabopt.controller_bfgs import design_bfgs, minimize_spectral_abscissa

options={'ftol': 1e-6, 'gtol': 1e-6}

sol = minimize_spectral_abscissa(plant, order=0, method="L-BFGS-B", options={"disp": True, **options}, callback=None, type = "barrier")

x = sol.x
K = x.reshape(K.shape)

# update the closed-loop with the optimized controller parameters
cl = tdcpy.ClosedLoop(plant, order=0, y_indices=[0,1,2], u_indices=[0], K0=K, hK=hK) 
cl_ddae = DDAE(E=cl.E,A=cl.A,hA=cl.hA)          # extracting the closed-loop ddae

cd, cdInfo = tdcpy.spectral_abscissa_diff(cl_ddae, return_info=True)
ssa_cl = strong_spectral_abscissa(cl_ddae, r=-1)

print(f"Optimized cd = {cd}")
print(f"Strong spectral abscissa of closed-loop with L-BFGS-B = {ssa_cl}")


# %%
# Plotting the roots of the closed-loop system after optimization with L-BFGS-B

region = [-0.5, 0.5, -100, 100]
cr_cl, cr_info = tdcpy.roots(cl_ddae, r=-1, discretization=200)

fig, ax = plt.subplots()
tdcpy.plot.eigen_plot(cr_cl, ax=ax)
ax.axvline(x=ssa_cl, color="red", linestyle="--")
ax.text(ssa_cl, 0, f"CD = {ssa_cl:.4f}", color="red", fontsize=10,
        verticalalignment="bottom", horizontalalignment="right")
title = ax.set_title("Closed-loop roots after optimization with L-BFGS-B")
ax.set_xlim(region[0], region[1])
ax.set_ylim(region[2], region[3])
plt.show()

# %%
#  Although the spectral abscissa is minimized successfully, we can achieve better
# Due to the non-smoothness of the strong spectral abscissa, the L-BFGS-B solver can get stuck in a local minima.
# In order to achieve better results, we can try a different solver for example Nelder-Mead,
# which is a derivative-free solver and repeat the process

sol = minimize_spectral_abscissa(plant, order=0, method="Nelder-Mead", options={"disp": True, **options}, callback=None, type = "barrier") 

x = sol.x
K = x.reshape(K.shape)

# update the closed-loop with the optimized controller parameters
cl = tdcpy.ClosedLoop(plant, order=0, y_indices=[0,1,2], u_indices=[0], K0=K, hK=hK) 
cl_ddae = DDAE(E=cl.E,A=cl.A,hA=cl.hA)          # extracting the closed-loop ddae

# let us again compute the strong spectral abscissa of the closed-loop
cd, cdInfo = tdcpy.spectral_abscissa_diff(cl_ddae, return_info=True)
ssa_cl = strong_spectral_abscissa(cl_ddae, r=-1)

print(f"Optimized cd with Nelder-Mead = {cd}")
print(f"Strong spectral abscissa of closed-loop with Nelder-Mead = {ssa_cl}")


# %% 
# Plotting the roots of the closed-loop system after optimization with Nelder-Mead

cr_cl, _ = tdcpy.roots(cl_ddae, r=-1, discretization=200)

fig, ax = plt.subplots()
tdcpy.plot.eigen_plot(cr_cl, ax=ax)
title = ax.set_title("Closed-loop roots after optimization with Nelder-Mead")
ax.set_xlim(region[0], region[1])
ax.set_ylim(region[2], region[3])
ax.axvline(x=ssa_cl, color="red", linestyle="--")
ax.text(ssa_cl, 0, f"CD = {ssa_cl:.4f}", color="red", fontsize=10,
        verticalalignment="bottom", horizontalalignment="right")
plt.show()

