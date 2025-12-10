"""

"""

import logging

import numpy as np

import tdspy as tds
import tdspy.plot as tdsplot

# Set up logging
import logging
logger = logging.getLogger("tdspy")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# setup for nicer outputs
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

# system G
m1 = 0.7
c1 = 7.47
Ag = np.array([[0, 1],
               [0, -c1/m1]])
Bg = np.array([[0],[1/m1]])
Cg = np.array([[1, 0]])
Dg = np.array([[0]])

# system F - TODO

placed_zero = -2.3844 + 1j* 23.7554

# controller parameters
omega_vector = 1.75 * np.pi * 2 * np.array([1,2,3.])
T = 0.45
N = 20 # number of delays and number of gains
tau_vector = np.linspace(0, T, 20)
gamma = 0.25
a_vector = np.array([0.0021, 0.5170, 1.1402, 1.3194, 1.0347, 0.4305, -0.2457, 
                     -0.7440, -0.9116, -0.7504, -0.4075, -0.0992, -0.0004, 
                     -0.1478, -0.4044, -0.5105, -0.2114, -0.0108, 0.0029, 
                     -0.0031]) # gains obtained by optimization process
# a_vector = np.array([0., 0, 0, 24.4114, -24.4114, 0., 0., 0., 0., 16.4248,
#                      -14.6090, -1.8157, 3.4909, -3.4909, 0, 0, 0, 0, 0, 0]) # gains obtained by optimization process

# controller parameters
Ac = np.array([[-4.2160, -0.4111],
               [-0.4111, -4.2160]])
Bc = np.array([[3.9145],[3.9145]])
Cc = np.array([[3.9145, 3.9145]])
Dc = np.array([[3.6454]])


# construct DDAE for shaper only and calculate zeros
E = np.array([[1.]])
A = np.zeros(shape=(1,1,1))
hA = np.array([0.])

B = a_vector[np.newaxis, np.newaxis, :]
hB = tau_vector

C = np.array([[[1.]]])
hC = np.array([0.])

D = np.array([[[gamma]]])
hD = np.array([0.])

ddae = tds.DDAE(A, hA, E=E, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)

z, _ = tds.zeros(ddae, r=[-10, 5, 0, 50])
print(z)

import matplotlib.pyplot as plt
tdsplot.eigen_plot(z)
plt.plot([np.real(placed_zero)],[np.imag([placed_zero])], "ro", alpha=0.4)

# Construct DDAE for inverse shaper
# x = [xs, v]
# dxs = SUM ak*v(t-tau_k)
# 0 = v -1/gamma * xs + 1/gamma * z
E = np.array([[1., 0],[0, 0]])
A = np.zeros(shape=(2,2,N))
A[0, 1, 0] = -1./gamma
A[1, 1, 0] = 1.
A[0,1,:] = a_vector
hA = tau_vector

ddae = tds.DDAE(A, hA, E=E)

cr, cr_info = tds.roots(ddae, r=[-10, 5, 0, 50])
print(cr)






# Construct DDAE

# create DDAE
# E = np.zeros(shape=(8, 8))
# E[:2, :2] = np.eye(2) # system G
# E[6:8, 6:8] = np.eye(2) # controller

# print(f"{E=}")

# A = np.zeros(shape=(8, 8, N))

# A[:2, :2, 0] = Ag
# A[:2, 2, 0] = Bg[:,0]
# A[2, :2, 0] = Cg[0, :]
# A[2, 2, 0] = Dg[0,0]


# A[5:7, 5:7, 0] = Ac
# A[5:7, 7, 0] = -Bc[:,0]
# A[7, 5:7, 0] = Cc[0, :]
# A[7, 7, 0] = -Dc[0,0]

# A[2, 3, 0] = -1
# A[3, 3, 0] = -1
# A[3, 4, 0] = 1
# A[-1, 2, 0] = -1

# A[3, -1, 0] = gamma
# A[4,-1,:] = a_vector

# for i in range(A.shape[2]):
#     print(A[:,:,i])

# hA = tau_vector
# print(tau_vector)

# ddae = tds.DDAE(A, hA, E=E)

# cr, cr_info = tds.roots(ddae, r=[-10, 5, 0, 100], discretization=100)

# print(cr)
# plt.plot(np.real(cr), np.imag(cr), "b+", alpha=0.5)

# plt.show()






