"""
Example: Create closed-loop for the vibration control setup, interconnected by a static feedback controller of gain K

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

    B1  = [  0   -0.8511 0   0   0   0   0   1.9231  ]'

    C1  = [ 1   0   0   0   0   0   0   0   
            0   1   0   0   0   0   0   0
            0   0   0   0   1   0   0   0
            0   0   0   0   0   1   0   0
            0   0   0   0   0   0   1   0
            0   0   0   0   0   0   0   1   ]

    """
    
    A0 = np.array([[    0,      1,  0,      0,      0,      0,   0,         0           ],
                    [   -2156.6,-6, 637.4,  0.7,    320.9,  0,   346.4,     1.5         ],
                    [   0,      0,   0,     1,      0,      0,   0,         0           ],
                    [   1483.2,  1.7, -2891.1, -5.3,    1407.9,     3.7,    0,  0       ],
                    [   0,   0,   0,   0,   0,   1,   0,   0                            ],
                    [   517.1,   0,   975.3,   2.5, -2795.6, -9.3,    0,    0           ],
                    [   0,   0,   0,   0,   0,   0,   0,   1                            ],
                    [   782.7,   3.5,   0,   0,   0,   0,   -782.7, -3.5                ]])
    A = np.stack([A0], axis=2)

    hA = np.array([0.])


    B2 =  np.array([[   0,   0,   0,   0,   0,   1.3717,  0,   0   ]]).T    
    B1 =  np.array([[   0,   -0.8511,   0,   0,   0,   0,     0,   1.9231   ]]).T
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

def generate_controller() -> tds.DDAE:
    """ generates static output feedback controller according to the paper
    Dc = [  ]
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

    # zeros = tdspy.zeros(rdde, r=[-2, 1, -60, 60], input_index=1, output_index=6)
    
    cont = generate_controller()