"""
Example: Create closed-loop for the vibration control setup, interconnected by a static feedback controller of gain K

[1] Saldanha, A., H. Silm, W. Michiels, and T. Vyhlidal(2022). “An Optimization-Based Algorithm for Simultaneous 
    Shaping of Poles and Zeros for Non-Collocated Vibration Suppression”. In: IFAC-PapersOnLine 55.16, pp. 394–399.
"""

import numpy as np
import tdspy as tds
import tdspy.controller
import tdspy.plot
from tdspy.common.composition import concatenate_2x2_by_delays
from tdspy.common.compress import compress_matrices_delays

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

    hA = np.array([0.0])

    B1 =  np.array([[   0,   0,   0,   0,   0,   0,     0,   0   ],
                    [   0,   0,   0,   0,   0,   0,     0,   0   ],
                    [   0,   0,   0,   0,   0,   1/m2,  0,   0   ]]).T    
    B2 =  np.array([[   0,   -1/m0,   0,   0,   0,   0,     0,   1/ma], # u1 column
                    [   0,   1/m0,   0,   0,   0,   0,     0,   0  ], # u2 column
                    [   0,   0,   0,   0,   0,   0,     0,   0   ]]).T # d column
                    
    B = np.stack([B1, B2], axis=2)
    hB = np.array([0.0, 0.002])
    C1 = np.array([[ 1,   0,   0,   0,   0,   0,   0,   0   ],
                   [ 0,   1,   0,   0,   0,   0,   0,  0    ],
                   [ 0,   0,   0,   0,   1,   0,   0,   0   ],
                   [ 0,   0,   0,   0,   0,   1,   0,   0   ],
                   [ 0,   0,   0,   0,   0,   0,   1,   0   ],
                   [ 0,   0,   0,   0,   0,   0,   0,   1   ],
                   [ 0,   0,   1,   0,   0,   0,   0,   0   ]]) # the last row is z
    C = np.stack([C1], axis=2)
    hC = np.array([0])
    D = np.zeros(shape=(7,3,1), dtype=float)
    hD = np.array([0.])

    rdde = tdspy.DDAE(A=A, hA=hA, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)
    # rdde = tdspy.RDDE(A=A, hA=hA,B=B, hB = hB, C = C, hC=hC)

    return rdde

def generate_controller1() -> tds.DDAE:
    """ generates static output feedback controller according to the paper
    Dc = [  ]
    """

    controller = tdspy.controller.create_static_controller(
        K = np.array([[-523.50, 9.93,  617.88, -8.61, 144.06, -7.73]])
    )

    return controller

def generate_controller() -> tds.DDAE:
    """ generates static output feedback controller according to the paper
    Dc = [  ]
    """

    # controller = tdspy.controller.create_static_controller(
    #     K = np.array([[-523.50, 9.93,  617.88, -8.61, 144.06, -7.73]])
    # )

    controller = tdspy.controller.create_static_controller(
        K = np.array([[ -1.031,     25.11,      0.898,      4.73,       -348.19,    -7.69   ],
                      [ 0.542,      -23.02,     -.0798,     -52.14,     1.88,       -15.50 ]])
    )

    return controller

def generate_controller_0() -> tdspy.DDAE:
    """ generates dynamic controller of 0 order using output feedback single-input controller
    according to the article TDS2024
    xc'(t)  = Ac xc(t) + Bc1 y(t-tau1) + Bc2 y(t-tau2) + Bc3 y(t-tau3) + Bc4 y(t-tau4)
    u(t)    = Cc xc(t) + Dc1 y(t-tau1) + Dc2 y(t-tau2) + Dc3 y(t-tau3) + Dc4 y(t-tau4)

    Here,
    K       =   [           |                                   ]
                    -----------------------------------------
                [           |   Dc1     Dc2     Dc3     Dc4     ]
    """


    Ac = np.zeros(shape=(1,1,0))
    hA = np.array([])

    Bc = np.zeros(shape=(1,4,0))
    hB = np.array([])

    Cc = np.zeros(shape=(1,1,0))
    hC = np.array([])

    Dc1 = np.array([[-0.2313, -0.9674, -0.000, -863.274]])
    Dc2 = np.array([[0., -66.3, -0.1, 1008.3]])
    Dc3 = np.array([[0., -6.5,0.,1853.5]])
    Dc4 = np.array([[-0., -228.3, -0.1, 2776.6]])

    D = np.stack([Dc1,Dc2,Dc3,Dc4],axis=2)
    hD = np.array([0.05, 0.10, 0.15, 0.20])

    ddae = tdspy.DDAE(A=Ac, hA=hA, B=Bc, hB=hB, C=Cc, hC=hC, D=D, hD=hD)

    return ddae

    # controller = tdspy.controller.create_dynamic_controller(A,B,C,D)
    # return controller


def generate_controller_2() -> tdspy.DDAE:
    """ generates dynamic controller of first order using output feedback single-input controller
    according to the article TDS2024
    xc'(t)  = Ac xc(t) + Bc1 y(t-tau1) + Bc2 y(t-tau2) + Bc3 y(t-tau3) + Bc4 y(t-tau4)
    u(t)    = Cc xc(t) + Dc1 y(t-tau1) + Dc2 y(t-tau2) + Dc3 y(t-tau3) + Dc4 y(t-tau4)

    Here,
    K       =   [       Ac  |   Bc1     Bc2     Bc3     Bc4     ]
                    -----------------------------------------
                [       Cc  |   Dc1     Dc2     Dc3     Dc4     ]
    """


    Ac = np.array([[-0.2313]])
    A = np.stack([Ac], axis=2)
    hA = np.array([0.])

    Bc1 = np.array([[0.000011871026690,  -0.000001633967929,  -0.096619515615447, 0.009609940786724]])
    Bc2 = np.array([[0.000012655720309,  -0.000001745203638,  -0.097102568003455, 0.009658695494486]])
    Bc3 = np.array([[-0.000008124456882,   0.000969071451097,  -0.097528596846931, 0.006345809570310]])
    Bc4 = np.array([[0.000079469623068,  -0.003649620510399,  -0.098032218002247,  0.015131687528009]])
    
    B = np.stack([Bc1,Bc2,Bc3,Bc4], axis=2)
    hB = np.array([0.05, 0.10, 0.15, 0.20])

    C = np.stack([np.array([[0.9674]])],axis=2)
    hC = np.array([0.])

    Dc1 = np.array([[-863.276410540652,  -66.322974500701,   1008.250387805891, -6.515309874083]])
    Dc2 = np.array([[1853.498088494986,  -228.288496302348,   2776.561842823067, -93.571222552822]])
    Dc3 = np.array([[ -2073.565017148369,  -327.260607372574,  -566.810081824136, -164.661906556694]])
    Dc4 = np.array([[-4100.968918319806,  -87.763270193528,  -2812.964755539708, -53.638395252897]])

    D = np.stack([Dc1,Dc2,Dc3,Dc4], axis=2)
    hD = np.array([0.05, 0.10, 0.15, 0.20])

    ddae = tdspy.DDAE(A=A, hA=hA, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)

    _, K, hK = concatenate_2x2_by_delays(np.eye(1), A, B, C, D, hA, hB, hC, hD)
    K, hK = compress_matrices_delays(K, hK)

    return ddae, K, hK

    # controller = tdspy.controller.create_dynamic_controller(A,B,C,D)
    # return controller



if __name__ == "__main__":
    # Set up logging
    import logging
    logger = logging.getLogger("tdspy")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    rdde = generate_system() # system we want to stabilitze

    # create closed loop representation
    controller, K, hK = generate_controller_2()
    cl = tdspy.ClosedLoop(rdde, 1, [0,1,4,5], [0], K0=K, hK=np.array([0.0, 0.05, 0.10, 0.15, 0.20]))

    # roots of original system, controller and closed loop
    cr_system, _ = tdspy.roots(cl.system, r=-10)
    cr_controller, _ = tdspy.roots(cl.controller, r=-30)
    cr_cl, _ = tdspy.roots(cl, r=-10)

    # zeros closed loop
    zr_cl, _ = tdspy.zeros(cl, r=[-10,2,0,200], input_index=0, output_index=0)

    import tdspy.plot
    import matplotlib.pyplot as plt

    # tdspy.plot.eigen_plot(cr)
    # plt.show()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    
    ax1.set_title("system")
    tdspy.plot.eigen_plot(cr_system, ax=ax1)

    ax2.set_title("closed loop")
    tdspy.plot.eigen_plot(cr_cl, ax=ax2)

    ax3.set_title("controller")
    tdspy.plot.eigen_plot(cr_controller, ax=ax3)

    ax4.set_title("closed loop zeros")
    tdspy.plot.eigen_plot(zr_cl, ax=ax4)
    
    plt.show()


    # zeros, zeros_info = tdspy.zeros(rdde, r=[-2, 1, -60, 60], input_index=1, output_index=6)
    
    # # cont = generate_controller()
    # # system = tdspy.controller.interconnect(rdde, cont, [0,1,2,3,4,5], [0,1,2,3,4,5], [0,1], [0,1])

    # # cont = generate_controller1()
    # # system = tdspy.controller.interconnect(rdde, cont, [0,1,2,3,4,5], [0,1,2,3,4,5], [0], [0])

    # cont = generate_controller_2()
    # system = tdspy.controller.interconnect(rdde, cont, [0,1,4,5], [0,1,2,3], [0], [0])


    # cr, cr_info = tdspy.roots(rdde, r=-10)
    # print(f"rightmost root of the ol is {np.max(np.real(cr))}")

    # system.is_essentially_retarded 
    # cr, cr_info = tdspy.roots(system, r=-10)
    # print(f"rightmost root of the cl is {np.max(np.real(cr))}")
    # print(cr)

    # zr, zr_info = tdspy.zeros(system, r=[-10,2,0,200], input_index=-1, output_index=-1)
    # print(zr)


    
