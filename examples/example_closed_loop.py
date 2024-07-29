"""
Example: Create closed-loop for the vibration control setup, interconnected by a static feedback controller of gain K

[1] Saldanha, A., H. Silm, W. Michiels, and T. Vyhlidal(2022). “An Optimization-Based Algorithm for Simultaneous 
    Shaping of Poles and Zeros for Non-Collocated Vibration Suppression”. In: IFAC-PapersOnLine 55.16, pp. 394–399.
"""

import numpy as np
import tdspy as tds
import tdspy.controller
import tdspy.plot

def generate_system() -> tds.RDDE:
    """ generates rdde for the system described in the article
    x'(t) = A x(t) + B2 f(t) + B1 u(t-tau)
    y(t) = C2 x(t)
    z(t) = C1 x(t)

    with 
    A = [   0   1   0   0   0   0   0   0;
            -2156.6 -6  637.4   0.7 320.9   0   346.4;  
            0   0   0   1   0   0   0   0;
            1483.2  1.7 -2891.1 -5.3    1407.9;
            0   0   0   0   0   1   0   0;
            517.1   0   975.3   2.5 -2795.6 -9.3    0;  
            0   0   0   0   0   0   0   1;
            782.7   3.5 0   0   0   0   0   -782.7      ]
    
    B = [   0   0   0   0   0   1.3717  0   0   ]'

    C = [   0   0   1   0   0   0   0   0   ]

    D = 0

    B1  = [     0   -0.8511 0   0   0   0   0   1.9231  ]'

    C1  = [     1   0   0   0   0   0   0   0   
                0   1   0   0   0   0   0   0
                0   0   0   0   1   0   0   0
                0   0   0   0   0   1   0   0
                0   0   0   0   0   0   1   0
                0   0   0   0   0   0   0   1       ]

    """
    
    # Masses

    m0 = 1.1750
    m1 = 0.5050
    m2 = 0.7290
    ma = 0.5200

    # Stiffness
    k0 = 1001
    k1 = 749
    k2 = 711
    k3 = 950
    ka = 407
    k4 = 377

    # Damping
    ca = 1.8000
    c0 = 4.3500
    c1 = 0.8500
    c2 = 1.8500
    c3 = 4.9500
    c4 = 0

    # Entering matrix entries

    a21 = -(k0+k1+ka+k4)/m0
    a22 = -(c0+c1+ca+c4)/m0
    a23 = k1/m0
    a24 = c1/m0
    a25 = k4/m0
    a26 = c4/m0
    a27 = ka/m0
    a28 = ca/m0

    a41 = k1/m1
    a42 = c1/m1
    a43 = -(k1+k2)/m1
    a44 = -(c1+c2)/m1
    a45 = k2/m1
    a46 = c2/m1
    a47 = 0
    a48 = 0

    a61 = k4/m2
    a62 = c4/m2
    a63 = k2/m2
    a64 = c2/m2
    a65 = -(k2+k3+k4)/m2
    a66 = -(c2+c3+c4)/m2
    a67 = 0
    a68 = 0

    a81 = ka/ma
    a82 = ca/ma
    a83 = 0
    a84 = 0
    a85 = 0
    a86 = 0
    a87 = -ka/ma
    a88 = -ca/ma


    A0 = np.array([[    0,      1,      0,      0,      0,      0,      0,     0           ],
                    [   a21,    a22,    a23,    a24,    a25,    a26,    a27,   a28         ],
                    [   0,      0,      0,      1,      0,      0,      0,     0           ],
                    [   a41,    a42,    a43,    a44,    a45,    a46,    a47,   a48         ],
                    [   0,      0,      0,      0,      0,      1,      0,      0          ],
                    [   a61,    a62,    a63,    a64,    a65,    a66,    a67,    a68        ],
                    [   0,      0,      0,      0,      0,      0,      0,      1          ],
                    [   a81,   a82,     a83,    a84,    a85,    a86,   a87,     a88        ]])
    A = np.stack([A0], axis=2)

    hA = np.array([0.])


    B2 =  np.array([[   0,   0,   0,   0,   0,   1/m2,  0,   0   ]]).T    
    B1 =  np.array([[   0,   -1/m0,   0,   0,   0,   0,     0,   1/ma   ]]).T
    B = np.stack([B1, B2], axis=1)
    hB = np.array([0.0])
    C1 = np.array([[ 1,   0,   0,   0,   0,   0,   0,   0   ],
                   [ 0,   1,   0,   0,   0,   0,   0,  0    ],
                   [ 0,   0,   0,   0,   1,   0,   0,   0   ],
                   [ 0,   0,   0,   0,   0,   1,   0,   0   ],
                   [ 0,   0,   0,   0,   0,   0,   1,   0   ],
                   [ 0,   0,   0,   0,   0,   0,   0,   1   ],
                   [ 0,   0,   1,   0,   0,   0,   0,   0   ]]) # the last row is z
    C = np.stack([C1], axis=2)
    hC = np.array([0])
    D = np.zeros(shape=(7,2,1), dtype=float)
    hD = np.array([0.])

    rdde = tdspy.DDAE(A=A, hA=hA, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)
    # rdde = tdspy.RDDE(A=A, hA=hA,B=B, hB = hB, C = C, hC=hC)

    return rdde

def generate_controller01() -> tds.DDAE:
    """ generates static output feedback single-input controller according to the paper
    Dc = [  144.06  -7.73   617.88  -8.61   -523.5  9.93    ]
    """
    A = np.zeros(shape=(1,1,0))
    hA = np.zeros(shape=(0,))
    B = np.zeros(shape=(1,6,0))
    hB = np.zeros(shape=(0,))
    C = np.zeros(shape=(1,1,0))
    hC = np.zeros(shape=(0,))
    D = np.array([[144.06, -7.73, 617.88, -8.61, -523.50, 9.93]])
    D = np.stack([D], axis=2)
    hD = np.array([0.])

    ddae = tdspy.DDAE(A=A, hA=hA, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)

    return ddae

def generate_controller02() -> tds.DDAE:
    """ generates static output feedback multi-input controller according to the paper
    Dc =    [  -348.19  -7.69   0.898   4.73    -1.031  25.11       ]
            [   1.88    -15.50  -0.798  -52.14  0.542   -23.02      ]
    """
    A = np.zeros(shape=(1,1,0))
    hA = np.zeros(shape=(0,))
    B = np.zeros(shape=(1,6,0))
    hB = np.zeros(shape=(0,))
    C = np.zeros(shape=(2,1,0))
    hC = np.zeros(shape=(0,))
    D = np.array([[     -348.19,    -7.69,      0.898,      4.73,       -1.031,     25.11       ],
                  [     1.88,       -15.50,     -0.798,     -52.14,     0.542,      -23.02      ]])
    D = np.stack([D], axis=2)
    hD = np.array([0.])

    ddae = tdspy.DDAE(A=A, hA=hA, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)

    return ddae



if __name__ == "__main__":
    # Set up logging
    import logging
    logger = logging.getLogger("tdspy")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    rdde = generate_system()

    zeros = tdspy.zeros(rdde, r=[-2, 1, -60, 60], input_index=1, output_index=6)
    
    cont01 = generate_controller01()        # single-input controller

    cont02 = generate_controller02()        # multi-input controller