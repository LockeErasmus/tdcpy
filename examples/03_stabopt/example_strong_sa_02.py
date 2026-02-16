"""
Example 3.2 - strong stabilization
==================================

We consider ...

TODO
"""
import numpy as np
import matplotlib.pyplot as plt

import tdspy
import tdspy.plot

# %%
#
#
#
#%%
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

plant = tdspy.DDAE(
    A=[A0], hA=[0],
    B=[Bu], hB=[5.],
    C=[C], hC=[0],
    D=[D1, D2], hD=[2.5, 5],
)

print(plant)

# static controller
K = np.array([[     0.0409,     0.0612,     0.3837  ]])
hK = np.array([0])
closed_loop = tdspy.ClosedLoop(plant, order=0, y_indices=[0, 1, 2], u_indices=[0], K0=np.stack([K], axis=2), hK=hK)
cl_ddae = tdspy.DDAE(E=closed_loop.E, A=closed_loop.A, hA=closed_loop.hA)


# %%
cl_roots, rootsInfo = tdspy.roots(closed_loop, r=[-1, 1, -100, 100])
cl_roots, rootsInfo = tdspy.roots(cl_ddae, r=[-1, 1, -100, 100])
print(f"mas RE of roots of closed-loop: {np.max(np.real(cl_roots))}, Adrian value= -0.0309 ")
# print(f"SA: {tdspy.spectral_abscissa(cl_ddae, r=-0.1)}, Adrian value= -0.0309")

tdspy.utils.print_system_matrices(cl_ddae)

tdspy.plot.eigen_plot(cl_roots)
plt.show()
# print(f"Roots of closed-loop: {tdspy.spectral_abscissa(closed_loop, r=-0.1)}")    # must be = -0.0309



# cd,cdInfo = tds.spectral_abscissa_diff(cl_dde)                             # must be = -0.0308894, more or less correct
# print(f"CD of diff: {cd}")
# gamma, gammaInfo = tds.gamma(cl_ddae,r=0,is_compressed=0)   
# print(f"gamma0 of diff: {gamma}")
# # gamma_norm_diff, gamma_norm_diff_Info = gamma_normalized_diff(cl_dde.A[:,:,1:], cl_dde.hA[1:], r=0, is_compressed=1)  
# g, info = gamma_normalized_diff(DD, hDD, 0, correction=True,is_compressed=0) 
# print(f"gamma_norm_diff of diff: {g}")