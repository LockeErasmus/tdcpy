"""
Example 2.3 from the TDS-CONTROL manual
We consider a quasi-poly nomial: 
    h(lambda) = lambda^2 + omega^2 - k exp(-lambda x 0.1)

This can be represented by the QP:
    QP = Sum p_j (lambda) exp(-lambda tau_i)

                                        [   1       ]                       [   1       ]
    i.e. qp =   [ omega^2   0   1   ]   [   lambda  ] + [ k     0   0   ]   [ lambda    ] exp(-lambda tau1)
                                        [   lambda^2]                       [lambda ^ 2 ]
    In MATLAB:
        qp = tds_create_qp([1 0 omega^2; 0 0 -k],[0 tau])

    In PYTHON:
                m-1                    n
        QP(s) =  SUM exp(-delays[i]*s) SUM coefs[i,j] * s**j
                i=0                   j=0

        
"""
import numpy as np
import tdcpy
import tdcpy.plot

from tdcpy.stability.characteristic_roots import rightmost_root

from scipy import linalg
from tdcpy.common.quasipoly import compress_qp, qp_to_ndde

# Set up logging
tdcpy.init_logger(level="DEBUG")

omega=2.
k=3.
tau=0.1

coeffs = np.array([[omega**2, 0., 1.],[-k, 0., 0.]])
delays = np.array([0.,tau])

A, hA, H, hH = qp_to_ndde(coeffs,delays,ascending=True)

ndde = tdcpy.NDDE(A=A,hA=hA,H=H,hH=hH)
ddae = ndde.to_ddae()

cr, RootInfo = tdcpy.roots(ndde,r=-2)



# plt = tdcpy.plot.eigen_plot(cr)
# ax = plt.axis

import matplotlib.pyplot as plt
plt.plot([np.real(cr)], [np.imag(cr)], "bo", alpha=0.25)

plt.show()

 