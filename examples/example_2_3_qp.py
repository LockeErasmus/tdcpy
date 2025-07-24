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
import tdspy
import tdspy.plot

from tdspy.stability.characteristic_roots import rightmost_root

# Set up logging
tdspy.init_logger(level="DEBUG")

# Create qp representation

p1 = np.array([[3, 3, 3, 3],
               [0, -1.5, 0, 0],
               [0, 0, 3, -5],
               [0, 5, 5, 5]])
A = np.stack([A0, A1], axis=2)
hA = np.array([0,1.])