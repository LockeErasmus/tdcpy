# Example 2.11 from the TDS-CONTROL manual
# We will compute the transmission zeros of the system defined by
# x'(t) = A0 x(t) + A1 x(t-1) + B u(t)
# y(t)  = C  x(t)
# with 
#       [   1   0   1   ]           [   0   1   0   ]       [   0   ]
# A0 =  [   0   0   0   ]   A1 =    [   0   0   0   ]   B = [   1   ]
#       [   0   1   0   ]           [   0   0   0   ]       [   0   ]
#
#  C =  [   1   0   0   ]
# in the rectangular region [-4 4] x 1j*[-50 50].
""" Example zeros

"""

import logging
import numpy as np
import tdspy as tds
import tdspy.plot

# Set up logging
import logging
logger = logging.getLogger("tdspy")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# enter matrices

A0 = np.array([[1,0,1],
              [0,0,0],
              [0,1,0]])
A1 = np.array([[0,1,0],
               [0,0,0],
               [0,0,0]])
A = np.stack([A0,A1],axis = 2)
hA = np.array([0, 1.])

B = np.zeros(shape=(3,1,1))
B[:,:,0] = np.array([[0,1,0]]).T
hB = np.array([0])

C = np.zeros(shape=(1,3,1))
C[:,:,0] = np.array([[1,0,0]])
hC = np.array([0])

D = np.zeros(shape=(1,1,1))
hD = np.array([0.])


# create ddae
ddae = tds.DDAE(A=A,hA=hA,B=B,hB=hB,C=C,hC=hC,D=D,hD=hD)

# compute zeros in given region
zr, zr_info = tds.zeros(ddae, r=[-4, 4, -50, 50])

print(zr)

# import matplotlib.pyplot as plt
# tdspy.plot.eigen_plot(zr)
# plt.show()

import matplotlib.pyplot as plt
tdspy.plot.eigen_plot(zr)
plt.show()

