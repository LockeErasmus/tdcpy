"""

"""

import logging

import numpy as np

import tdcpy as tds

# Set up logging
import logging
logger = logging.getLogger("tdcpy")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# system parameters
mp = 2.5
kp = 1500
cp = 2.0
ma = 0.5
ka = 500
ca = 2.0

# controller parameters
omega_vector = 1.75 * np.pi * 2 * np.array([1,2,3.])
T = 0.4
N = 20 # number of delays and number of gains
tau_vector = np.linspace(0, T, 20)

# obtain controller gains
# gains_vector

A = np.zeros(shape=(2*tau_vector.shape[0], N))
b = np.zeros(shape=(2*tau_vector.shape[0],))

for i, omega in enumerate(omega_vector):
    lhs = np.exp(-1j*tau_vector)
    lhs_re = np.real(lhs)
    lhs_im = np.imag(lhs)
    rhs_re = - ma*omega**2 + ka
    rhs_im = ca*omega

    A[2*i, :] = lhs_re
    A[2*i + 1, :] = lhs_im
    b[2*i] = rhs_re
    b[2*i + 1] = rhs_im

gains_vector = np.ravel(np.linalg.pinv(A)@ b[:, np.newaxis])

# create system dynamics
Ag = np.array([[0, 1, 0, 0],
              [-(kp+ka)/mp, -(cp+ca)/mp, ka/mp, ca/mp],
              [0, 0, 0, 1],
              [ka/ma, ca/ma, -ka/ma, -ca/ma]])
Bu = np.array([[0],[-1/mp],[0],[1/ma]])
Bf = np.array([[0],[1/mp],[0],[0]])
# B = np.concatenate([Bu, Bf], axis=1)
Cxp  = np.array([[1, 0, 0, 0]])
Cxa  = np.array([[0, 0, 1, 0]])
Cdxa = np.array([[0, 0, 0, 1]])

# create controller
# TODO


# create DDAE
E = np.zeros(shape=(5, 5))

A = np.zeros(shape=(5, 5, N))
E[:-1, :-1] = np.eye(4)
A[:-1, :-1, 0] = Ag
A[:-1, -1, 0] = Bu[:,0]
A[-1, -1, 0] = -1
A[-1, 2, :] = gains_vector
hA = tau_vector

B = np.zeros(shape=(5, 1, 1))
B[:-1, 0, 0] = Bf[:,0]
hB = np.array([0.])

C = np.zeros(shape=(2, 5, 1))
C[0, 0, 0] = 1 # xp(t)
C[1, 2, 0] = 1 # xa(t)
hC = np.array([0.])

D = np.zeros(shape=(2,1,1))
hD = np.array([0.])

ddae = tds.DDAE(A, hA, E=E, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)


# compute zeros in given region

zr, zr_info = tds.zeros(ddae, r=[-10, 10, 0.0, 100])

print(zr)


