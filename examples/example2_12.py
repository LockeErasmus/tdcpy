"""
Example 2.12 from the TDS-CONTROL manual
Let the plant be given by:
    H(s)    = H0(s) e^(-s tau)

The Smith predictor is given by:
    C'(s) = C(s) / (1 + C H0 - C H0 e^(-s tau))

Consider the delay mismatch problem: 
              1                 s 
    H0 =  --------  ,   C(s) = --- + 2,
            s + 1               2

The closed-loop system is given by:
    CL(s) = H0(s) C(s) / (1 + H0(s) C(s)) e^(-s tau)

We want to determine the regions in the (tau, delta) space for
which the closed-loop system is stable.

The quasipolynomial is given by:
    QP(s) = 3/2 lambda + 3 + (lambda/2 + 2) exp(-lambda tau) 
                + (-lambda/2 - 2) exp(-lambda (tau + delta)).

This can be represented by the QP:
    QP = Sum p_j (lambda) exp(-lambda tau_i)

                                [   1       ]                     [   1       ]
    i.e. qp =   [ 3   3/2   ]   [   lambda  ] + [ 2     1/2   ]   [ lambda    ] exp(-lambda tau1)

                                                    [   1   ]
                                + [ -2    -1/2  ]   [ lambda ] exp(-lambda tau2)
    In MATLAB:
        qp = tds_create_qp([1.5 3;0.5 2;-0.5 -2],[0 tau tau+delta])

    In PYTHON:
                m-1                    n
        QP(s) =  SUM exp(-delays[i]*s) SUM coefs[i,j] * s**j
                i=0                   j=0
        
"""


import numpy as np
import tdspy as tds
<<<<<<< HEAD
import tdspy.plot as plt
=======
import matplotlib.pyplot as plt
>>>>>>> adrian_test
from tdspy.common.quasipoly import qp_to_ndde
from tdspy.common.quasipoly import compress_qp, qp_to_ndde


tau, delta = 1., 0.5

coeffs = np.array([[3.,1.5],[2.0,0.5],[-2.0,-0.5]])
delays = np.array([0.,tau,tau+delta])

A, hA, H, hH = qp_to_ndde(coeffs,delays,ascending=True)

ndde = tds.NDDE(A=A,hA=hA,H=H,hH=hH)

<<<<<<< HEAD
print(ndde)

print(ndde.A)
print(ndde.H)
=======
ndde.print()

tau_grid = np.linspace(0,8,201)
delta_grid = np.linspace(-8,10,451)

# Z = np.zeros((len(delta_grid), len(tau_grid)))

# for i2 in range(0,len(tau_grid)-1):
#     tau = tau_grid[i2]
#     for i1 in range(0,len(delta_grid)):
#         delta = delta_grid[i1]
#         if np.abs(tau+delta)<1e-8:
#             # case: tau+delta = 0
#             if np.abs(tau)<1e-8:
#                 tau = 1e-8
#             hH[0],hH[1] = tau, 1e-8
#             hA[1],hA[2] = tau, 1e-8
#             ndde = tds.NDDE(H=H,hH=hH,A=A,hA=hA)
#             Z[i1,i2] = tds.strong_spectral_abscissa(ndde)
#         elif tau+delta<=0.1:
#             # case: "real" delay cannot be negative
#             Z[i1,i2] = -np.inf
#         else:
#             hH[0], hH[1] = tau, tau+delta
#             hA[1], hA[2] = tau, tau+delta
#             ndde = tds.NDDE(H=H,hH=hH,A=A,hA=hA)
#             Z[i1,i2] = tds.strong_spectral_abscissa(ndde)

# X, Y = np.meshgrid(tau_grid,delta_grid)


# plt.contour(X,Y,Z,[0,0])
# plt.plot(plt.xlim,[0,0],'k-.')
# plt.show()
>>>>>>> adrian_test
