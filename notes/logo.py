import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tdcpy
import tdcpy.plot

import scipy.linalg as linalg
from tdcpy.common.discretization import discretize_ddae

# Create RDDE matrix representation
A0 = np.array([[-1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, -10, -4],
               [0, 0, 4, -10]])
A1 = np.array([[3, 3, 3, 3],
               [0, -1.5, 0, 0],
               [0, 0, 3, -5],
               [0, 5, 5, 5]])

rdde = tdcpy.RDDE(A=[A0, A1], hA=[0, 1])

# logo specific settings
y_lim = 50
x_lim = (-4, 0.8)

# discretize and solve EVP
Pi_N, Sigma_N = discretize_ddae(rdde.E, rdde.A, rdde.hA, discretization=100, s0=0)
raw_roots = linalg.eig(Sigma_N, Pi_N, left=False, right=False)
raw_roots = raw_roots[np.isfinite(raw_roots)] # get rid of inf and NaN
raw_roots = raw_roots[(np.abs(np.imag(raw_roots)) <= y_lim*0.9) & (np.real(raw_roots >= 0.9*x_lim[0]))]

fig, ax = plt.subplots(1,1, figsize=(3, 1))

ax.scatter(np.real(raw_roots), np.imag(raw_roots), marker="x",
            color="k")

Pi_N, Sigma_N = discretize_ddae(rdde.E, rdde.A, rdde.hA, discretization=10, s0=0)
raw_roots = linalg.eig(Sigma_N, Pi_N, left=False, right=False)

roots, info = tdcpy.roots(rdde, r=-1.0)
raw_roots = info.discretization_eigenvalues
raw_roots = raw_roots[(np.abs(np.imag(raw_roots)) <= y_lim*0.9) & (np.real(raw_roots >= 0.9*x_lim[0]))]

ax.scatter(np.real(raw_roots), np.imag(raw_roots), marker="o",
            color="k", facecolor="none")



ax.set_xlim(x_lim)
ax.set_ylim((-y_lim, y_lim))
ax.axis("off")
# ax.set_aspect("equal")

# plt.show()

fig.savefig("logo.svg", bbox_inches="tight", pad_inches=0, transparent=True)
#fig.savefig("logo.png", dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)