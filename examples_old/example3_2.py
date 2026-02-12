""" Example 3.2 taken from TDS-Manual
    
    x'(t)   =   A x(t)  +   B u(t-tau1)
    y(t)    =   C x(t)  +   D1 u(t-tau2)    +   D2 u(t-tau3)
    returns A, B, C with correct shapes

"""

import tdspy as tds
import numpy as np
from scipy import linalg
from tdspy import controller
from tdspy.common.delay_difference_equation import ddae_to_diff, normalize_diff
from tdspy.stability.gamma_r import gamma_normalized_diff
from tdspy.stabopt.controller_bfgs import find_feasible_point, minimize_spectral_abscissa
from tdspy.stabopt.utils import diff_dependency_mask
from tdspy.stabopt.gradients import grad_gamma0, gradient_test


A0 = np.array([
[   -0.08,  -0.03,  0.2     ],
[   0.2,    -0.04,  -0.005  ],
[   -0.06,  0.2,    -0.07    ],
])
A = np.stack([A0], axis=2)
hA = np.array([0.])
Bu = np.array([ [-0.1],[-0.2],[0.1]    ])
B = np.stack([Bu],axis=2)
hB = np.array([5.])
C = np.array(np.eye(3))
C = np.stack([C],axis=2)
hC = np.array([0.])
D1 = np.array([ [3.],[4.],[1.]    ])
D2 = np.array([ [0.4],[-0.4],[-0.4]])
D = np.stack([D1,D2],axis = 2)
hD = np.array([2.5,5.])
ddae = tds.ddae.DDAE(A=A,hA=hA,B=B,hB=hB,C=C,hC=hC,D=D,hD=hD)


from tdspy.stabopt.gradients import grad_gamma0, gradient_test


A0 = np.array([
        [   -0.08,  -0.03,  0.2     ],
        [   0.2,    -0.04,  -0.005  ],
        [   -0.06,  0.2,    -0.07    ],
    ])
A = np.stack([A0], axis=2)
hA = np.array([0])
Bu = np.array([ [-0.1],[-0.2],[0.1]    ])
B = np.stack([Bu],axis=2)
hB = np.array([5.])
C = np.array(np.eye(3))
C = np.stack([C],axis=2)
hC = np.array([0])
D1 = np.array([ [3],[4],[1]    ])
D2 = np.array([ [0.4],[-0.4],[-0.4]])
D = np.stack([D1,D2],axis = 2)
hD = np.array([2.5,5.])
ddae = tds.ddae.DDAE(A=A,hA=hA,B=B,hB=hB,C=C,hC=hC,D=D,hD=hD)

K = np.array([[     0.0409,     0.0612,     0.3837  ]])
K = np.array([[0.0365, 0.0480, 0.0416]])
K = np.array([[1.,2.,3.]])
hK = np.zeros(shape=(1,)) # no delay in the controller

# forming the closed-loop
K = K[:, :, np.newaxis]  # add delay axis
cl = tds.ClosedLoop(ddae, 0, [0,1,2], [0], K, hK=hK)
cl_ddae = tds.DDAE(E=cl.E, A=cl.A, hA=cl.hA)

# examining the stability
E,P,hP = cl.E, cl.A[:,:,:-1], cl.hA[:-1]
uE,vE = cl.uE, cl.vE
BB,CC = cl.BB, cl.CC

# examining the delay-difference equation
D,hD = ddae_to_diff(E,cl_ddae.A,cl_ddae.hA)
DD,hDD = normalize_diff(D,hD)
g0, gInfo = gamma_normalized_diff(DD, hDD, 0, correction=True, n_theta=10)
print(f"Original gamma0 = {g0}, gamma_normalized = {gInfo}")

# extract controller parameters affecting the delay-difference equation
Kmask = np.full_like(K, fill_value=True,dtype=bool)
x0 = K.reshape(-1)
p0 = K.reshape(-1)

from tdspy.stabopt.controller_bfgs import find_feasible_point
p_feasible = find_feasible_point(E, P, hP, K, hK, BB, CC, nstart=5, gamma0_threshold=0.5)

print(f"Feasible point found: {p_feasible.x}")





# sol = minimize_spectral_abscissa(ddae, order=0, method="L-BFGS-B", options={"disp": True}, callback=None)


