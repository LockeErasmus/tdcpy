"""
Strong stability
================

First, we consider Example 13 from :cite:`michiels2010spectrum`

.. math::

    \begin{bmatrix}
        1 & 0\\
        0 & 0
    \end{bmatrix}
    \dot{x}(t)
    =
    \begin{bmatrix}
        0 & -\frac{1}{8}\\
        -1 & 1
    \end{bmatrix}
    x(t)
    +
    \begin{bmatrix}
        0 & 0\\
        0 & a
    \end{bmatrix}
    x(t-\tau_1)
    +
    \begin{bmatrix}
        0 & 0\\
        0 & \frac{1}{2}
    \end{bmatrix}
    x(t-\tau_1)

and show that for certain values of :math:`a` the system is not strongly
stable. Next, we show how dradient of spectral abscissa of associated 
delay-difference equation can be computed and used to obtain strong stability.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

import tdspy
import tdspy.plot


E = np.array([
    [1, 0.],
    [0, 0],
])
A0 = np.array([
    [0, -1./8],
    [-1, 1],
])
A1 = np.array([
    [0, 0],
    [0, -0.25],
])
A2 = np.array([
    [0, 0],
    [0, 0.5],
])


ddae1 = tdspy.DDAE(A=[A0, A1, A2], hA=np.array([0, 1, 2]), E=E)
ddae1_perturbed = tdspy.DDAE(A=[A0, A1, A2], hA=np.array([0, 0.99, 2]), E=E)

A1[1,1] = -0.75
ddae2 = tdspy.DDAE(A=[A0, A1, A2], hA=np.array([0, 1, 2]), E=E)
ddae2_perturbed = tdspy.DDAE(A=[A0, A1, A2], hA=np.array([0, 0.99, 2]), E=E)

region = [-1, 0.5, -200, 200]
if False:
    fig, (ax1, ax2) = plt.subplots(1,2, sharex=True, sharey=True)

    # Solve a=0.25
    cr1, _ = tdspy.roots(ddae1, r=region)
    cr1_perturbed, _ = tdspy.roots(ddae1_perturbed, r=region)
    sa1 = tdspy.sa(ddae1)
    cd1, _ = tdspy.cd(ddae1)

    # solve a=0.75
    cr2, _ = tdspy.roots(ddae2, r=region)
    cr2_perturbed, _ = tdspy.roots(ddae2_perturbed, r=region)
    sa2 = tdspy.sa(ddae2)
    cd2, _ = tdspy.cd(ddae2)

    ax1.set_title("Case $a=0.25$")
    tdspy.plot.complex_scatter_axplot(cr1, ax=ax1, marker="+", color="green", label="nominal")
    tdspy.plot.complex_scatter_axplot(cr1_perturbed, ax=ax1, marker="o", edgecolor="blue", facecolor="none", label="perturbed")
    ax1.axvline(x=cd1, color='b', linestyle='--', alpha=0.5, label=r"$c_D$")
    ax1.axvline(x=sa1, color='r', linestyle='--', alpha=0.5, label=r"$\alpha$")
    ax1.set_xlabel(r"$\Re$")
    ax1.set_ylabel(r"$\Im$")
    ax1.set_xlim(region[0], region[1])
    ax1.set_ylim(region[2], region[3])
    ax1.legend()

    ax2.set_title("Case $a=0.75$")
    tdspy.plot.complex_scatter_axplot(cr2, ax=ax2, marker="+", color="green", label="nominal")
    tdspy.plot.complex_scatter_axplot(cr2_perturbed, ax=ax2, marker="o", edgecolor="blue", facecolor="none", label="perturbed")
    ax2.axvline(x=cd2, color='b', linestyle='--', alpha=0.5, label=r"$c_D$")
    ax2.axvline(x=sa2, color='r', linestyle='--', alpha=0.5, label=r"$\alpha$")
    ax1.set_xlabel(r"$\Re$")
    ax1.set_ylabel(r"$\Im$")
    ax2.set_xlim(region[0], region[1])
    ax2.set_ylim(region[2], region[3])
    ax2.legend()

    plt.show()
# %%
# Next, let us show how gradient of :math:`c_D` can be computed and used to
# minimize :math:`c_D` and obtain strong stability (assuming right most root is
# stable).
#
# We are going to proceed manually, i.e. follow preciselly the equations in TODO
#%%

# step one, obtain U and V
uE = linalg.null_space(E.T)
vE = linalg.null_space(E)

# step two, obtain D0, D1, D2, observe that D1 can be written by - a B C
B = np.array([[0],[1]])
C = np.array([[0,1]])

D0 =  uE.T @ A0 @ vE
D1 =  uE.T @ (-B @ C) @ vE # * a
D2 =  uE.T @ A2 @ vE

# and normalize, since by inverse of D0, note we can use inverse here, since D0 is nice, but in general we use LU decomposition
H1 = linalg.inv(D0) @ D1 # * a
H2 = linalg.inv(D0) @ D2
hH = np.array([1, 2])

# set a=0.25 and compute cd
from tdspy.stability.spectral_abscissa import spectral_abscissa_diff

a = 0.75
cd, cd_info = spectral_abscissa_diff(np.stack([H1 * a, H2], axis=2), hH)

# examine cd_info
print(cd_info, cd)

# %%
theta = cd_info.th # critical values of theta, 0. is prepended, such that dimensions check
u = cd_info.u
v = cd_info.v
s = cd_info.s

dM = (np.conj(u)[np.newaxis,:] @ H1*a @ v[:, np.newaxis] * hH[0] * np.exp(-cd * hH[0]) * np.exp(1j*theta[0])
      + np.conj(u)[np.newaxis,:] @ H2 @ v[:, np.newaxis] * hH[1] * np.exp(-cd * hH[1]) * np.exp(1j*theta[1]))
denum = np.real(np.conj(s) * dM / np.inner(np.conj(u), v))

uHv = np.conj(u)[np.newaxis,:] @ H1 @ v[:, np.newaxis] * np.exp(-cd * hH[0]) * np.exp(1j*theta[0])
num = np.real(np.conj(s) * uHv / np.inner(np.conj(u), v))

da = num / denum # derivative of cd w.r.t. a


print(da)

# %%
progress = {
    "a": [],
    "roots": [],
    "roots_perturbed": [],
    "cd": [],
}

print(a * B @ C)

for i in range(10):
    cd, cd_info = spectral_abscissa_diff(np.stack([H1 * a, H2], axis=2), hH)
    A1[1,1] = -a # a * B @ C
    cr, _ = tdspy.roots(tdspy.DDAE(A=[A0, A1, A2], hA=np.array([0, 1, 2]), E=E), r=region)
    crp, _ = tdspy.roots(tdspy.DDAE(A=[A0, A1, A2], hA=np.array([0, 0.99, 2]), E=E), r=region)
    progress["a"].append(a)
    progress["roots"].append(cr)
    progress["roots_perturbed"].append(crp)
    progress["cd"].append(cd)

    theta = cd_info.th # critical values of theta, 0. is prepended, such that dimensions check
    u = cd_info.u
    v = cd_info.v
    s = cd_info.s

    dM = (np.conj(u)[np.newaxis,:] @ H1*a @ v[:, np.newaxis] * hH[0] * np.exp(-cd * hH[0]) * np.exp(1j*theta[0])
        + np.conj(u)[np.newaxis,:] @ H2 @ v[:, np.newaxis] * hH[1] * np.exp(-cd * hH[1]) * np.exp(1j*theta[1]))
    denum = np.real(np.conj(s) * dM / np.inner(np.conj(u), v))

    uHv = np.conj(u)[np.newaxis,:] @ H1 @ v[:, np.newaxis] * np.exp(-cd * hH[0]) * np.exp(1j*theta[0])
    num = np.real(np.conj(s) * uHv / np.inner(np.conj(u), v))

    da = num / denum # derivative of cd w.r.t. a

    a -= 0.1 * np.ravel(da)[0] # TODO
    print(f"cd: {cd}, da: {da}, a: {a}")


fig, ax = plt.subplots()
s = ax.scatter([], [], marker="+", color="green", label="nominal")
sp = ax.scatter([], [], marker="o", edgecolor="blue", facecolor="none", label="perturbed")
ax.set_xlabel(r"$\Re (\lambda)$")
ax.set_ylabel(r"$\Im (\lambda)$")
legend = ax.legend()

ax.set_xlim(region[0], region[1])
ax.set_ylim(region[2], region[3])

def update(n):
    cr = progress["roots"][n]
    crp = progress["roots_perturbed"][n]
    s.set_offsets(np.column_stack([np.real(cr), np.imag(cr)]))
    sp.set_offsets(np.column_stack([np.real(crp), np.imag(crp)]))
    ax.axvline(x=progress["cd"][n], color='b', linestyle='--', alpha=0.5, label=r"$c_D$")
    # legend.get_texts()[1].set_text("Solution of EVP N={}".format(n))

import matplotlib.animation as animation
ani = animation.FuncAnimation(fig, update, frames=range(10), interval=500)

plt.show()




raise


print(B @ K0[:,:,0] @ C)

from tdspy.stabopt.gradients import func_cd
import tdspy
import tdspy.plot
import matplotlib.pyplot as plt

tdspy.init_logger(level="INFO")



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



