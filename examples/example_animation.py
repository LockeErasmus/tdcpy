"""
A simple example of an animated plot
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.linalg

import tdspy
import tdspy.common
import tdspy.common
import tdspy.ddae
import tdspy.roots

# Set up logging
import logging
logger = logging.getLogger("tdspy")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Create DDAE representation
A0 = np.array([[-1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, -10, -4],
               [0, 0, 4, -10]])
A1 = np.array([[3, 3, 3, 3],
               [0, -1.5, 0, 0],
               [0, 0, 3, -5],
               [0, 5, 5, 5]])
A = np.stack([A0, A1], axis=2)
hA = np.array([0,1.])
r = -2.5

rdde = tdspy.rdde.RDDE(A=A, hA=hA)


# initialize plot
# fig, ax = plt.subplots()

# r=np.array([0], dtype=np.complex128)
# ax.scatter(np.real(r), np.imag(r), marker="o",
#             edgecolors="r", facecolors='none', label="python")

# def animate(n):
#     s0 = 0
#     dae = tdspy.common.discretize(rdde, n, s0=s0)
#     r = scipy.linalg.eig(dae.A, dae.E, left=False, right=False)
#     r = r[np.isfinite(r)] - s0  # get rid of inf and NaN and shift back
#     ax.clear()
#     ax.scatter(np.real(r), np.imag(r), marker="o",
#                edgecolors="r", facecolors='none')
#     ax.set_xlim(-20,3)
#     ax.set_ylim(-1000,1000)
#     return ax

# # Init only required for blitting to give a clean slate.
# def init():
#     return ax

# ani = animation.FuncAnimation(fig, animate, np.arange(10, 20), init_func=init, interval=25, blit=True´,)
# plt.show()

fig, ax = plt.subplots()

s0 = 0
dae = tdspy.common.discretize(rdde, 200, s0=s0)
r = scipy.linalg.eig(dae.A, dae.E, left=False, right=False)
r = r[np.isfinite(r)] - s0  # get rid of inf and NaN and shift back
ax.scatter(np.real(r), np.imag(r), marker="x",
            color="b", label="location of roots")
ax.set_xlim(-6,2)
ax.set_ylim(-125, 125)
s = ax.scatter([], [], marker="o", edgecolors="r", facecolors='none', label="Solution of EVP")
ax.legend()

text = ax.text(0.05, 0.5, "N=", ha='left', va='top', transform=ax.transAxes)

def update(n):
    s0 = 0
    dae = tdspy.common.discretize(rdde, n+10, s0=s0)
    r = scipy.linalg.eig(dae.A, dae.E, left=False, right=False)
    r = r[np.isfinite(r)] - s0  # get rid of inf and NaN and shift back
    s.set_offsets(np.column_stack([np.real(r), np.imag(r)]))
    text.set_text(f"N={n+10}")
    
ani = animation.FuncAnimation(fig, update, frames=65, interval=100,)
writer = animation.PillowWriter(fps=15,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
#ani.save('scatter.gif', writer=writer)
plt.show()