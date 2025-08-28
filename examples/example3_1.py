# Defining the system
"""
DDAEs:
Consider the feedback interconnection of the form

    x'(t)   = A0 x(t) + A1 x(t-tauA1) + A2 x(t-tauA2) + A3 x(t-tauA3) + A4 x(t-tauA4) + A5 x(t-tauA5) + A6 x(t-tauA6)
                        + B11 u(t-tauB1)

    y(t)    = C1 x(t) 

with
1. Static output feedback
    u(t)    = Dc y(t)

2. Dynamic output feedback

    x'(t)   = Ac xc(t) + Bc y(t)

    u(t)    = Cc xc(t) + Dc y(t)

with

    A0,A1,A2,A3,A4,A5,A6,A7 defined in TDS-CONTROL
    B1, C1, defined in TDS-CONTROL
"""

import tdspy as tds
import numpy as np
from tdspy.stability.characteristic_roots import rightmost_root

Th, Ta,Td, Tc = 14, 3, 3, 25
Kb, Ka, Kd, Kc, Ku = 0.24, 1, 0.94, 0.81, 0.39
nh, tb, te, td = 6.5, 40, 13, 18
tc, nc, tu = 2.8, 9.2, 13.2

hA, hB = np.array([0, nh, tb, te, td, tc, nc]), np.array([tu])

A0 = np.zeros([5,5])
A0[1,0], A0[1,1] = Ka/Ta, (-Ka-1)/Ta
A0[2,2], A0[4,3] = -1/Td, -1
A1, A1[0,0]     = np.zeros([5,5]), -1/Th
A2, A2[0,1]     = np.zeros([5,5]), Kb/Th
A3, A3[1,3]     = np.zeros([5,5]), 1/Ta
A4, A4[2,1]     = np.zeros([5,5]), Kd/Td
A5, A5[3,2]     = np.zeros([5,5]), Kc/Tc
A6, A6[3,3]     = np.zeros([5,5]), -1/Tc
A = np.stack([A0,A1,A2,A3,A4,A5,A6],axis=2)

B0 = np.array([[Ku/Th], [0.], [0.], [0.], [0.]])
B = np.stack([B0],axis=2)

C0 = np.eye(5)
C = np.stack([C0],axis=2)

D = np.zeros(shape=(5,1,1), dtype=float)    # must be defined, otherwise error!
hD = np.array([0.])

hC = np.array([0.])

plant = tds.DDAE(A=A,hA=hA,B=B,hB=hB,C=C,hC=hC,D=D,hD=hD)
plant.print()

# Finding the rightmost root - Test 
z, z_info = rightmost_root(E=plant.E, A=plant.A, hA=plant.hA, r=2)
print(np.max(np.real(z)))


# Design a static output feedback controller of the form
# u = Dc y

import numpy as np
import tdspy as tds
from tdspy.stabopt.controller_bfgs import design_bfgs
from tdspy.stabopt.gradients import func_sa, gradient_test
from tdspy.common.composition import concatenate_2x2_by_delays
from tdspy.stabopt.gradients import func_sa, gradient_test
import tdspy.controller as controller

cont = controller.create_static_controller(np.array([[-0.1659, -0.2968, -0.3612, -0.3629, 0.0168]]))
cont.print()

E, K, hK = concatenate_2x2_by_delays(cont.E, cont.A, cont.B, cont.C, cont.D, cont.hA, cont.hB, cont.hC, cont.hD)
hK0 = np.array([0.])
cl = tds.ClosedLoop(plant,order=0,y_indices=[0,1,2,3,4],u_indices=[0],K0=K,hK=hK0)

print(cl.A.shape)
print(cl.hA.shape)
cl.print()

# the shape after forming the cl is incorrect, therefore, the below causes an error
cr_system, _ = tds.roots(cl.system, r=-0.1)
cr_cl, _ = tds.roots(cl, r=-1)


# E = cl.E
# P = cl._A
# hP = cl._hA
# K0 = np.zeros([1,5,1])
# hK = cl.hK
# B = cl.BB
# C = cl.CC

# sol = design_bfgs(cl.E, P, hP, K0, hK, B, C, options={"disp": True, "eps":0.1})

cont.D