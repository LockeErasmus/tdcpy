## Test example
# \dot{x} = A x + B1 u(t)
# y = C1 x
# z = C2 x
# u = K y(t-tau)


import numpy as np
import tdspy
import tdspy.plot
import tdspy.controller
from scipy.linalg import block_diag

from tdspy.stability.characteristic_roots import rightmost_root

# Set up logging
tdspy.init_logger(level="DEBUG")

# # System parameters
k = 100; m = 1; tau = 0.1; n = 2;
# Define system matrices
A0   = np.array([[0.,1.],
               [-k/m,0.]])
D   = np.array([[0.]])

# # Input-output matrices

p   = 1;  # inputs
q   = 1;  # outputs
B1  = np.array([[0.], 
               [1/m]]);     # input
hB1 = np.array([0.])

C1 = np.array([[0., 1.]]);       # output
hC1 = np.array([0.1])


A = np.stack([A0], axis=2)
hA = np.array([0.])

r = -10

# ###############PART 1#########################
# # Form rdde, ddae

# ddae = tdspy.DDAE(A=A, hA=hA,B1=B1,hB1=hB1,C1=C1,hC1=hC1)
# rdde = tdspy.RDDE(A=A, hA=hA,B1=B1,hB1=hB1,C1=C1,hC1=hC1)
# cr, info = tdspy.roots(rdde, r=r)

# z = np.max(np.real(cr))

# print(f"roots are {cr}")

# # Adam, please change this: Form closed-loop with K 

# controller = tdspy.controller.create_static_controller(
#         K = np.array([[0.5]])
#     )
# hK = np.array([0.])

# cl = tdspy.ClosedLoop(ddae, 0, [0], [0], K0=controller, hK=hK)

# cr2, info2 = tdspy.roots(cl, r=r)

# print(f"cl_roots are {cr}")




###############PART 2#########################

ddae_E = block_diag(np.eye(n), np.zeros((q, q)), np.zeros((q, q)), np.zeros((p, p)))

# A0
ddae_A0 = np.block([
    [A[:, :, 0],         B1,                np.zeros((n, q)),      np.zeros((n, q))],
    [np.zeros((q, n)),   np.zeros((q, p)),  np.zeros((q, q)),     -np.eye(q)],
    [C1,                 np.zeros((q, p)), -np.eye(q),             np.zeros((q, q))],
    [np.zeros((p, n)),  -np.eye(p),         np.zeros((p, q)),      np.zeros((p, q))]
])

# A1
ddae_A1 = np.block([
    [np.zeros((n, n)),     np.zeros((n, p)), np.zeros((n, q)),     np.zeros((n, q))],
    [np.zeros((q, n)),     np.zeros((q, p)), np.eye(q),            np.zeros((q, q))],
    [np.zeros((q, n)),     np.zeros((q, p)), np.zeros((q, q)),     np.zeros((q, q))],
    [np.zeros((p, n)),     np.zeros((p, p)), np.zeros((p, q)),     np.zeros((p, q))]
])

tau = np.array([0.1])
ddae_hA = np.array([0, 0.1]);

# B1
ddae_B1 = np.vstack([
    np.zeros((n, p)),
    np.zeros((q, p)),
    np.zeros((q, p)),
    np.eye(p)
])
ddae_hB1 = np.array([0.])

# C1
ddae_C1 = np.hstack([
    np.zeros((q, n)),
    np.zeros((q, p)),
    np.zeros((q, q)),
    np.eye(q)
])
ddae_hC1 = np.array([0.])

# ddae
A = np.stack([ddae_A0,ddae_A1],axis=2)
B = np.stack([ddae_B1],axis=2)
C = np.stack([ddae_C1],axis=2)
ddae2 = tdspy.DDAE(A=A, hA=ddae_hA,B1=B,hB1=ddae_hB1,C1=C1,hC1=ddae_hC1)
# ol2 = tds_create_ddae(ddae_E,{ddae_A0, ddae_A1},ddae_hA,{ddae_B1},0, ...
#                             {ddae_C1},0);


print(f"isretarded: {ddae2.is_essentially_retarded}")