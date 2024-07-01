"""
NDDE example from MATLAB manual page 16, equation (2.20)

"""

import numpy as np
import tdspy as tds
import tdspy.plot
from scipy.linalg import block_diag

# Set up logging
import logging
logger = logging.getLogger("tdspy")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Plant parameters

m0=1.1750; m1=0.5050; m2=0.7290; ma=0.5200;                     # mass

k0=1001; k1=749; k2=711; k3=950; ka=407; k4=377;                # stiffness

ca=1.8000; c0=4.3500; c1=0.8500; c2=1.8500; c3=4.9500; c4=0;    # damping

# System matrices

A = np.array([  [           0,                  1,           0,           0,               0,               0,          0,      0       ],
                [   -(k0+k1+ka+k4)/m0, -(c0+c1+ca+c4)/m0,       k1/m0,       c1/m0,          k4/m0,          c4/m0,     ka/m0,  ca/m0   ],
                [           0,                  0,           0,           1,               0,               0,          0,      0       ],
                [       k1/m1,              c1/m1, -(k1+k2)/m1, -(c1+c2)/m1,           k2/m1,           c2/m1,          0,      0       ],
                [           0,                  0,           0,           0,               0,               1,          0,      0       ],
                [       k4/m2,              c4/m2,           k2/m2,       c2/m2,    -(k2+k3+k4)/m2, -(c2+c3+c4)/m2,     0,      0       ],
                [           0,                  0,           0,           0,               0,               0,          0,      1       ],
                [       ka/ma,              ca/ma,           0,           0,               0,               0,          -ka/ma, -ca/ma  ]]);
hA = np.array([0]);                                             # delay vector corresponding to A0

B2 = np.array([[  0,   0,   0,   0,  0,  1/m2,   0,  0  ]]).T;      # system input matrix
hB2 = np.array([0]);        # exogenous input delay

C2 = np.array([[  0,  0,   1,   0,   0,   0,   0,   0   ]]);         # system output matrix
hC2 = np.array([0]);        # system output delay

D = np.array([0]);                                              # feedthrough matrix

B1 = np.array([[  0,   -1/m0,   0,   0,   0,  0,    0 ,  1/ma   ]]).T;   # input matrix
hB1 = np.array([0]);        # input delay

C1 = np.array( [[   1,   0,   0,   0,   0,   0,   0,   0    ],
                [   0,   1,   0,   0,   0,   0,   0,   0    ],
                [   0,   0,   0,   0,   0,   0,   1,   0    ],
                [   0,   0,   0,   0,   0,   0,   0,   1    ]]);    # feedback output matrix
hC1 = np.array([0]);        # output delay

n = A.shape[0];             # system order
nu = B1.shape[1];           # number of inputs
ny = C1.shape[0];           # number of outputs

# Create open-loop ddae

ol = tdspy.DDAE(A=np.stack([A],axis=2), hA=hA)


# Controller

Ny = 2;         # no of delays
nyd = ny*Ny;    # no of outputs x no of delays
Nu = 0;         # no of input delays
nud = nu*Nu;    # no of inputs x no of delays
nc = 0;         # order of the dynamic controller

# Creating a DDAE from the above
n_ddae = n+nyd+nc+nu;   # order of the ddae

E = block_diag(np.eye(n),np.zeros((nyd,nu)),np.eye(nc),np.zeros((nu,nyd)));  # E matrix

# Construct matrix A0
A0 = np.block([[     A,                     B1,                     np.zeros((n,nc)),       np.zeros((n,nyd))       ],
              [     np.zeros((nyd,n)),      np.zeros((nyd,nu)),     np.zeros((nyd,nc)),     -np.eye((nyd))          ],
              [     np.zeros((nc,n)),       np.zeros((nc,nu)),      np.zeros((nc,nc)),      np.zeros((nc,nyd))      ],
              [     np.zeros((nu,n)),       -np.eye((nu)),          np.zeros((nu,nc)),      np.zeros((nu,nyd))      ]]);  

# # Construce matrix A1
# A1 = np.block([[    np.zeros((n,n)),        B1,                     np.zeros((n,nc)),       np.zeros((n,nyd))       ],
#                [    np.zeros((nyd,n)),      np.zeros((nyd,nu)),     np.zeros((nyd,nc)),     np.zeros((nyd,nyd))     ],
#                [    np.zeros((nc,n)),       np.zeros((nc,nu)),      np.zeros((nc,nc)),      np.zeros((nc,nyd))      ],
#                [    np.zeros((nu,n)),       np.zeros((nu,nu)),      np.zeros((nu,nc)),      np.zeros((nu,nyd))      ]] );

A = np.stack([A0], axis=2)
hA = np.array([0])

# Construct matrix H1, H2
H1 = np.block([[    np.zeros((n,n)),        np.zeros((n,nu)),       np.zeros((n,nc)),       np.zeros((n,nyd))       ],
               [    C1,                     np.zeros((4,nu)),       np.zeros((4,nc)),       np.zeros((4,nyd))     ],
               [    np.zeros((4,n)),        np.zeros((4,nu)),       np.zeros((4,nc)),       np.zeros((4,nyd))     ],
               [    np.zeros((nc,n)),       np.zeros((nc,nu)),      np.zeros((nc,nc)),      np.zeros((nc,nyd))      ],
               [    np.zeros((nu,n)),       np.zeros((nu,nu)),      np.zeros((nu,nc)),      np.zeros((nu,nyd))      ]] );

H2 = np.block([[    np.zeros((n,n)),        np.zeros((n,nu)),       np.zeros((n,nc)),       np.zeros((n,nyd))       ],
               [    np.zeros((4,n)),        np.zeros((4,nu)),       np.zeros((4,nc)),       np.zeros((4,nyd))     ],
               [    C1,                     np.zeros((4,nu)),       np.zeros((4,nc)),       np.zeros((4,nyd))     ],
               [    np.zeros((nc,n)),       np.zeros((nc,nu)),      np.zeros((nc,nc)),      np.zeros((nc,nyd))      ],
               [    np.zeros((nu,n)),       np.zeros((nu,nu)),      np.zeros((nu,nc)),      np.zeros((nu,nyd))      ]] );

H = np.stack([H1,H2,], axis=2)

# Three set of delays
hH = np.array([0.1, 0.4]);

ndde = tds.NDDE(A=A, hA=hA, H=H, hH=hH)

ddae = ndde.to_ddae().compress()

r = [-0.1, 1, -100, 100]
cr, info = tdspy.roots(ddae, r=r, max_size_evp=1500)

import matplotlib.pyplot as plt
tdspy.plot.eigen_plot(cr)
plt.show()

