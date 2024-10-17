"""
Example 3.3 taken from TDS-Manual
            [   1.25    -0.8    -0.95   ]       +   [   2   ]
x'(t)   =   [   0.175   -0.4    -0.125  ] x(t)      [   0   ] u(t)
            [   -1.15   -0.4    0.65    ]           [   -2  ]

y(t)    =   [   -7  25  -11     ] x(t)  + 1 u(t)

and the stabilizing static output feedback law

u(t)    =  - 5 y(t)
"""
import numpy as np
import matplotlib.pyplot as plt
import tdspy as tds

def create_system1() -> tds.ddae:
    """ 
    returns A, B, C with correct shapes
    """
    A = np.array([
        [   1.25,   -0.8,   -0.95   ],
        [   0.175,  -0.4,   -0.125  ],
        [   -1.15, -0.4,   0.65    ],
    ])
    Bu = np.array([ [2],[0],[-2]    ])
    C = np.array([  [ -7, 25, -11 ]   ])
    D = [1]
    ddae = tds.ddae(E=np.eye(3),A=A,B=Bu,C=C,D=D)
    return ddae


def create_system2() -> tds.ddae:
    """ 
    returns A, B, C with correct shapes
    """
    A0 = np.array([
        [   -0.08,  -0.03,  0.2     ],
        [   0.2,    -0.04,  -0.005  ],
        [   -0.06,  0.2,    -0.07    ],
    ])
    A = np.stack([A0], axis=2)
    hA = np.array([0])
    Bu = np.array([ [-0.1],[-0.2],[0.1]    ])
    B = np.stack([Bu],axis=1)
    hB = np.array([5.])
    C = np.array(np.eye(3))
    C = np.stack([C],axis=2)
    hC = np.array([0])
    D1 = np.array([ [3],[4],[1]    ])
    D2 = np.array([ [0.4],[-0.4],[-0.4]])
    D = np.stack([D1,D2],axis = 2)
    hD = np.array([2.5,5.])
    ddae = tds.ddae.DDAE(A=A,hA=hA,B=B,hB=hB,C=C,hC=hC,D=D,hD=hD)
    return ddae
 

def create_system3() -> tds.ddae:
    """ 
    % We will design a stabilizing acceleration feedback controller for the
    mass-spring system with an artificially introduced feedback delay
    x'(t) = [  0     1 ] x(t) + [ 0 ] u(t-tau)
            [-k/m   0]         [1/m]
    y(t) = [0 1] x'(t)
    with k = 100 N/m, m = 1kg and tau = 0.1 s.

    """
    k = 100; m = 1; tau = 0.1;
# Define transformed system matrices
    Ea = np.array([ [1, 0, 0],
                    [0, 1, 0],
                    [0, 1, 0]])
    Aa = np.array([[0,1,0],
                   [-k/m,0,0],
                   [0,0,1]])
    Ba = np.array([[0],[1/m],[0]])
    Ca = np.array([[0,0,1]])
    D = np.array([[1]])
    
    # Use tds_create_ddae to create a TDS_SS-representation
    
    ddae = tds.ddae(E=Ea,A=Aa,B=Ba,hB=np.array([tau]), C=Ca, D=D)

    return ddae

def create_cl_ddae() -> tuple:
    """
    
    """
    A, B, C, D = create_system1() # create system with defaults

if __name__ == "__main__":
    # Set up logging
    import logging
    logger = logging.getLogger("tdspy")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    ddae = create_system2()



    print(f"The DDAE is essentially neutral={ddae.is_essentially_neutral}")

    cd = tds.cd(ddae)

    print(f"strogn spectral abscissa of associated DIFF {cd=}")

