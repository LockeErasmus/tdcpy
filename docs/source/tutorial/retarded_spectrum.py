
import numpy as np
import tdcpy as tds
import matplotlib.pyplot as plt
from tdcpy.plot.eigenvalues import eigen_plot

A0 = np.array([[1, 0], [0, 1]], dtype=float)
A1 = np.array([[0, 1], [0.5, 0]], dtype=float)
delays = np.array([0,0.5])
rdde = tds.RDDE(A = [A0, A1], hA=delays)

roots, info = tds.roots(rdde, r=-10.0)
ax = eigen_plot(roots, title="Characteristic roots of the retarded system")
plt.show()