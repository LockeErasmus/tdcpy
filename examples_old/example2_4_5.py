""" 
Example 2.4: Computing the characteristic roots of a neutral DDE

Consider the NDDE:

    x'(t) = [   -0.6    -0.45   ] x(t)  +   [   -0.15   0.075   ] x(t-1)    +   [   3       -1.5    ] x'(t-1)
            [   0.1     -1.2    ]           [   0.225   -0.75   ]               [   2.5     -1      ]

Example 2.5: Relation between the characteristic roots of a neutral DDE and the characteristic roots of its underlying delay difference equation

The associated delay difference equation is given by

    x(t) +  [   3       -1.5    ] x(t-1)    =   0
            [   2.5     -1      ]

How do we obtain the delay difference equation from the above?

"""

import numpy as np
import tdcpy as tds
import tdcpy.plot

tds.init_logger(level="WARNING")    

A0 = np.array([[    -0.6,   -0.45   ],
                [    0.1,    -1.2   ]])
A1 = np.array([[    -0.15,  0.075   ],
                [   0.225,  -0.75   ]])
A = np.stack([A0,   A1],    axis=2)
hA = np.array([0.,  1.])

H = np.stack([np.array([[3,-1.5],[2.5,-1]])],axis=2)
hH = np.array([1.])

ndde = tds.NDDE(A=A,hA=hA,H=H,hH=hH)

ddae = ndde.to_ddae()

cr, RootsInfo = tds.roots(ddae,r=[-3,1,-60,60])

import matplotlib.pyplot as plt
tdcpy.plot.eigen_plot(cr)



#-------------------Beginning of Example 2.9----------------------#

dde = ddae.get_delay_difference_equation()

crd, RootsInfo = tds.roots(dde,r=[-3,1,-60,60])
crd

tdcpy.plot.eigen_plot(crd)


plt.plot([np.real(crd)], [np.imag(crd)], "bo", alpha=0.25)
plt.show()