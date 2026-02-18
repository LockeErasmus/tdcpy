import numpy as np
import tdcpy as tds
A0 = np.array([[0.25]])
A1 = np.array([[0.75]])
hA = np.array([0.,2.])
H1 = np.array([[-0.75]])
H2 = np.array([[0.5]])
hH = np.array([1.,2.])
ndde = tds.NDDE(H = [H1,H2], hH = hH, A = [A0, A1], hA=hA)

alpha = tds.spectral_abscissa(ndde)
CD = tds.strong_spectral_abscissa(ndde)

diff = ndde.get_delay_difference_equation()
CD_diff,_ = tds.spectral_abscissa_diff(diff)

import matplotlib.pyplot as plt
from tdcpy.plot.eigenvalues import eigen_plot

fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)

roots, info = tds.roots(ndde, r=[-1,2,-500,500], max_size_evp=1500)
eigen_plot(roots, ax=ax1, title="\tau_2=2")
ax1.axvline(x=CD_diff, color='red', linestyle='--', label=f'x = {CD_diff}')

# change hH[1] to 2.02
ndde.hH[1] = 2.02
roots, info = tds.roots(ndde, r=[-1,2,-500,500], max_size_evp=1500)
eigen_plot(roots, ax=ax2, title="\tau_2=2.02")
ax2.axvline(x=CD_diff, color='red', linestyle='--', label=f'x = {CD_diff}')

plt.show()

