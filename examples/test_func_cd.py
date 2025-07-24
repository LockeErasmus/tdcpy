## Testing the func_cd function
## Check with the example ex2_neutral from Ch3_DesigningStabilizingControllers

import numpy as np
import matplotlib.pyplot as plt
import tdspy as tds
import tdspy.controller
from tdspy.stabopt.gradients import gradient_test, func_cd
from numpy.linalg import inv
from tdspy.common.composition import concatenate_2x2_by_delays
from tdspy.stability.gamma_r import gamma_normalized_diff, gamma_diff
from tdspy.common.delay_difference_equation import ddae_to_diff, normalize_diff
from tdspy.stability.characteristic_roots import rightmost_root, roots_ddae
from tdspy.stability.spectral_abscissa import spectral_abscissa_diff, spectral_abscissa


def create_system() -> tds.ddae:
    """ 
    Example 3.2 taken from TDS-Manual
    x'(t)   =   A x(t)  +   B u(t-tau1)
    y(t)    =   C x(t)  +   D1 u(t-tau2)    +   D2 u(t-tau3)
        returns A, B, C with correct shapes
    """
    E = np.array([
        [   1,  0,  0,  0,  0,  0,  0   ],
        [   0,  1,  0,  0,  0,  0,  0   ],
        [   0,  0,  1,  0,  0,  0,  0   ],
        [   0,  0,  0,  0,  0,  0,  0   ],
        [   0,  0,  0,  0,  0,  0,  0   ],
        [   0,  0,  0,  0,  0,  0,  0   ],
        [   0,  0,  0,  0,  0,  0,  0   ],
    ])

    A0 = np.array([
        [   -0.08,  -0.03,  0.2,    0,  0,  0,  0   ],
        [   0.2,    -0.04,  -0.005, 0,  0,  0,  0   ],
        [   -0.06,  0.2,    -0.07,  0,  0,  0,  0   ],
        [   1,      0,      0,      -1,  0,  0, 0   ],
        [   0,      1,      0,      0,  -1,  0, 0   ],
        [   0,      0,      1,      0,  0,  -1, 0   ],
        [   0,      0,      1,      0,  0,  0,  1   ],
    ])
    A1 = np.array([
        [   0,  0,  0,  0,  0,  0,  0   ],
        [   0,  0,  0,  0,  0,  0,  0   ],
        [   0,  0,  0,  0,  0,  0,  0   ],
        [   0,  0,  0,  0,  0,  0,  3   ],
        [   0,  0,  0,  0,  0,  0,  4   ],
        [   0,  0,  0,  0,  0,  0,  1   ],
        [   0,  0,  0,  0,  0,  0,  0   ],
    ])
    A2 = np.array([
        [   0,  0,  0,  0,  0,  0,  -0.1    ],
        [   0,  0,  0,  0,  0,  0,  -0.2    ],
        [   0,  0,  0,  0,  0,  0,  0.1     ],
        [   0,  0,  0,  0,  0,  0,  0.4     ],
        [   0,  0,  0,  0,  0,  0,  -0.4    ],
        [   0,  0,  0,  0,  0,  0,  -0.4    ],
        [   0,  0,  0,  0,  0,  0,  0       ],
    ])
    A3 = np.array([
        [   0,  0,  0,  0,  0,  0,  0   ],
        [   0,  0,  0,  0,  0,  0,  0   ],
        [   0,  0,  0,  0,  0,  0,  0   ],
        [   0,  0,  0,  0,  0,  0,  0   ],
        [   0,  0,  0,  0,  0,  0,  0   ],
        [   0,  0,  0,  0,  0,  0,  0   ],
        [   0,  0,  0,  0,  0,  0,  0   ],
    ])

    A = np.stack([A0,A1,A2,A3], axis=2)
    hA = np.array([0., 2.5, 5.0, 0.])

    ddae = tds.ddae.DDAE(E=E,A=A,hA=hA)
    return ddae

def print_ddae(ddae: tds.DDAE):
    with np.printoptions(precision=4, linewidth=1000, suppress=True):
        print(f"A 4x4 matrix")
        print(ddae.E)
        print("-"*50)
        for i in range(ddae.mA):
            print(f"A[:,:,{i} - tau={ddae.hA[i]}")
            print(ddae.A[:,:,i])
            print("-"*50)

if __name__ == "__main__":
    # Set up logging
    import logging
    logger = logging.getLogger("tdspy")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create ddae, controller
    ddae = create_system()
    controller = tds.controller.create_static_controller(
        K = np.array([[     0.0409,     0.0612,     0.3837  ]])
    )

    D, hD = ddae_to_diff(E=ddae.E, A=ddae.A, hA=ddae.hA, uE=ddae.uE, vE=ddae.vE,is_compressed=False)
    diff = ddae.get_delay_difference_equation()
    print_ddae(diff)


    DD, hDD = normalize_diff(diff.A, diff.hA)





    