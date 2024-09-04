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

def create_system() -> tuple:
    """ 
    returns A, B, C with correct shapes
    """
    A = np.array([
        [   1.25,   -0.8,   -0.95   ],
        [   0.175,  -0.4,   -0.125  ],
            [-1.15, -0.4,   0.65    ],
    ])
    Bu = np.array([ [2],[0],[-2]    ])
    C = np.array([  [ -7, 25, -11 ]   ])
    D = [1]
    return A, Bu, C, D

def create_cl_ddae() -> tuple:
    """
    """
    A, B, C, D = create_system() # create system with defaults

if __name__ == "__main__":
    # Set up logging
    import logging
    logger = logging.getLogger("tdspy")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    ddae = create_cl_ddae()

    print(f"The DDAE is essentially neutral={ddae.is_essentially_neutral}")

    cd = tds.cd(ddae)

    print(f"strogn spectral abscissa of associated DIFF {cd=}")

