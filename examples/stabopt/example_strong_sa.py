"""
TODO
"""
import numpy as np
from scipy import linalg

E = np.array([
    [1, 0.],
    [0, 0],
])
P0 = np.array([
    [0, -1./8],
    [-1, 1],
])
P1 = np.array([
    [0, 0],
    [0, 0.5],
])
P = np.stack([P0, P1], axis=2)
hP = np.array([0, 2])

a = 1./4
a = 3./4
K0 = np.array([[[a]]])
hK = np.array([0.99])
# hK = np.array([1.0])
B = np.array([[0],[1]])
C = np.array([[0,1]])

print(B @ K0[:,:,0] @ C)

from tdspy.stabopt.gradients import func_cd
import tdspy
import tdspy.plot
import matplotlib.pyplot as plt

tdspy.init_logger(level="INFO")

uE = linalg.null_space(E.T)
vE = linalg.null_space(E)

print(uE)
print(vE)

stepsize = 1e-2
K = np.copy(K0)
for i in range(100):
    dK = func_cd(K.reshape(-1), E, P, hP, np.full_like(K0, fill_value=True), hK, B, C, uE, vE)

    K -= stepsize * dK

ddae_0 = tdspy.DDAE(A=[P0, P1, B @ K0[:,:,0] @ C], hA=np.r_[hP, hK], E=E)
ddae_star = tdspy.DDAE(A=[P0, P1, B @ K[:,:,0] @ C], hA=np.r_[hP, hK], E=E)

region = [-1, 1, -200, 200]

cd_star, _ = tdspy.cd(ddae_star)
cr_star, _ = tdspy.roots(ddae_star, r=region)

cd_0, _ = tdspy.cd(ddae_0)
cr_0, _ = tdspy.roots(ddae_0, r=region)


fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)

tdspy.plot.eigen_plot(cr_0, ax=ax1)
ax1.axvline(x=cd_0, color='r', linestyle='--', alpha=0.5)
ax1.set_title("nominal")

tdspy.plot.eigen_plot(cr_star, ax=ax2)
ax2.axvline(x=cd_star, color='r', linestyle='--', alpha=0.5)
ax2.set_title("optimized")

plt.show()



