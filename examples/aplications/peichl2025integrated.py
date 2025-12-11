r"""
Non-collocated vibration absorbtion via delayed resonator
=========================================================

In this :cite:`peichl2025integrated` authors considered chain of 3 flexibly
linked masses (first and last also connected to rigid frame). Harmonic
disturbance :math:`d(t) = \text{cos}(\omega t)` acts on last mass, absorber
:math:`m_a, k_a, c_a` together with voice coil :math:`u(t)` is mounted on
the first mass to supress harmonic disturbance.

The whole system can be described via 4 second order linear equations

.. math::

    M \ddot{x}(t) + C\dot{x}(t) + Kx(t) = B_u u(t) + B_d d(t),

where :math:`x=[x_a, x_1, x_2, x_3]` is a vector of displacement,
:math:`M=\text{diag}(m_a, m_1, m_2, m_3)` is mass matrix, :math:`C, K` 
are damping and stiffness matrices, respectively, defined as

.. math::

    C =
     \begin{bmatrix}
         c_a & -c_a & 0 & 0 \\
         -c_a & c_1 + c_2 + c_a & -c_2 & 0\\
         0 & -c_2 & c_2 + c_3 & -c_3 \\
         0 & 0 & -c_3 & c_3 + c_4
     \end{bmatrix}
     K =
     \begin{bmatrix}
         k_a & -k_a & 0 & 0 \\
         -k_a & k_1 + k_2 + k_a & -k_2 & 0 \\
         0 & -k_2 & k_2 + k_3 & -k_3 \\
         0 & 0 & -k_3 & k_3 + k_4
     \end{bmatrix},

input matrices are defined as

.. math::

    B_u =
     \begin{bmatrix}
         1 \\ -1 \\ 0 \\ 0
     \end{bmatrix}
     B_d =
     \begin{bmatrix}
         0 \\ 0 \\ 0 \\ 1
     \end{bmatrix}.

Control law is position delayed resonater parametrized via gain :math:`g` and
delay :math:`\tau`

.. math::

    u(t) = g ~ E_a^T x(t-\tau),
    
vector :math:`E_a^T = [1, 0, 0, 0]` encodes position of absorber mass.

Given the nominal parameters :cite:`peichl2025integrated` construct the closed
loop as retarded delay differential equations (RDDE) and check poles and zeros.
Closing the loop and writing and going from second order to first order yields
retarded system with two matrices :math:`A_0, A_1` and delays :math:`0, \tau`

.. math::

    A_0 =
     \begin{bmatrix}
        O_{4} & I \\
        -M^{-1} K & -M^{-1} C
     \end{bmatrix}
    A_1 = g
     \begin{bmatrix}
         o_4 \\ M^{-1} B_u
     \end{bmatrix}
     \begin{bmatrix}
         o_4  & E_a
     \end{bmatrix},

.. bibliography::
"""

import matplotlib.pyplot as plt
import numpy as np
import tdspy
import tdspy.plot

# Main chain of masses-springs-dampers
m1, m2, m3 = 1.49, 0.509, 1.1 # masses [kg]
k1, k2, k3, k4 = 1001, 749, 711, 950 # stiffness [Nm-1]
c1, c2, c3, c4 = 4.35, 0.85, 1.85, 4.95 # damping [Nsm-1]

# absorber - mass [kg], stiffness [Nm-1], damping [Nsm-1]
ma, ka, ca = 0.420, 407, 1.8

# control law - nominal case
g, tau = -78.05282, 0.03303
omega_target = 8.4 # [Hz]

# Modal matrices
M = np.diag([ma, m1, m2, m3])
K = np.array([
    [ka, -ka, 0, 0],
    [-ka, k1+k2+ka, -k2, 0],
    [0, -k2, k2+k3, -k3],
    [0, 0, -k3, k3+k4],
])
C = np.array([
    [ca, -ca, 0, 0],
    [-ca, c1+c2+ca, -c2, 0],
    [0, -c2, c2+c3, -c3],
    [0, 0, -c3, c3+c4],
])

Minv = np.linalg.inv(M)
A0 = np.block([
    [np.zeros(shape=(4,4)), np.eye(4)],
    [-Minv @ K, -Minv @ C],
])
Du = np.array([[1.], [-1], [0], [0]])
Dd = np.array([[0.], [0], [0], [1]])

Bu = np.block([
    [np.zeros(shape=(4,1))],
    [Minv @ Du],
])
Bd = np.block([
    [np.zeros(shape=(4,1))],
    [Minv @ Dd],
])
Ea = np.array([[1], [0], [0], [0], [0], [0], [0], [0],])
Et = np.array([[0], [0], [1], [0], [0], [0], [0], [0],])

A1 = g * Bu @ Ea.T

ddae = tdspy.RDDE(A=[A0, A1], hA=[0, tau], B=[Bd], hB=[0], C=[Et.T], hC=[0], D=[np.array([[0.]])], hD=[0.])
cr, info = tdspy.roots(ddae, r=-350)
z, _ = tdspy.zeros(ddae, r=[-350, 5, -4500, 4500])

fig, ax1 = plt.subplots(1,1)

ax1.scatter(np.real(z), np.imag(z), marker="o", facecolors='none', edgecolors='blue', alpha=0.8, label="zeros")
tdspy.plot.eigen_plot(cr, ax=ax1)
ax1.scatter([0, 0], [omega_target*np.pi, -omega_target*np.pi], marker="x", color="black", label=rf"$\omega=-j{omega_target}\pi$")
ax1.set_xlim((-340, 1))
ax1.set_ylim((3000, -3000))
ax1.legend(loc='lower right')

# add second detail plot to show area around origin
ax2 = fig.add_axes([0.55, 0.55, 0.3, 0.3])
ax2.scatter(np.real(z), np.imag(z), marker="o", facecolors='none', edgecolors='blue', alpha=0.8)
tdspy.plot.eigen_plot(cr, ax=ax2)
ax2.scatter([0, 0], [omega_target*np.pi, -omega_target*np.pi], marker="x", color="black")
ax2.set_xlim((-4, 1))
ax2.set_ylim((100, -100))
ax2.set_xlabel('')
ax2.set_ylabel('')

plt.show()

