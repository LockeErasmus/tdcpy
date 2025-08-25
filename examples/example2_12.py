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
import tdspy.plot as plt
from tdspy.common.quasipoly import qp_to_ndde
from tdspy.common.quasipoly import compress_qp, qp_to_ndde


tau, delta = 1., 0.5

coeffs = np.array([[3.,1.5],[2.0,0.5],[-2.0,-0.5]])
delays = np.array([0.,tau,tau+delta])

A, hA, H, hH = qp_to_ndde(coeffs,delays,ascending=True)

ndde = tds.NDDE(A=A,hA=hA,H=H,hH=hH)

print(ndde)

print(ndde.A)
print(ndde.H)