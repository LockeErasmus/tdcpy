"""
Example: Create closed-loop for the vibration control setup, interconnected by a static feedback controller of gain K

"""

import numpy as np
import tdspy as tds
import tdspy.controller
import tdspy.plot

from tdspy.common.composition import concatenate_2x2_by_delays
from tdspy.stability.characteristic_roots import rightmost_root, RightmostRootInfo

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

    B1  = [  0   -0.8511 0   0   0   0   0   1.9231  ]'

    C1  = [ 1   0   0   0   0   0   0   0   
            0   1   0   0   0   0   0   0
            0   0   0   0   1   0   0   0
            0   0   0   0   0   1   0   0
            0   0   0   0   0   0   1   0
            0   0   0   0   0   0   0   1   ]

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

    hA = np.array([0.])

    B1 =  np.array([[   0,   0,   0,   0,   0,   0,     0,   0   ],
                    [   0,   0,   0,   0,   0,   0,     0,   0   ],
                    [   0,   0,   0,   0,   0,   1/m2,  0,   0   ]]).T    
    B2 =  np.array([[   0,   -1/m0,   0,   0,   0,   0,     0,   1/ma], # u1 column
                    [   0,   1/m0,   0,   0,   0,   0,     0,   0  ], # u2 column
                    [   0,   0,   0,   0,   0,   0,     0,   0   ]]).T # d column
                    
    B = np.stack([B1, B2], axis=2)
    hB = np.array([0.0, 0.0019])
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
        K = np.array([[-1.031, 25.11, 0.898, 4.73, -348.19, -7.69],
                      [ 0.542, -23.02, -.0798, -52.14, 1.88, -15.50]])
    )

    return controller

def generate_controller_2() -> tdspy.DDAE:
    """ generates dynamic controller of first order using output feedback single-input controller
    x'(t)   = Ac x(t) + Bc y(t)
    u(t)    = Cc x(t) + Dc y(t)
    """

    Ac = np.array([[-0.2313]])
    A = np.stack([Ac], axis=2)
    hA = np.array([0.])

    Bc1 = np.array([[0., -0., -0.0966, 0.0096]])
    Bc2 = np.array([[0., -0., -0.0971, 0.0097]])
    Bc3 = np.array([[-0., 0.001, -0.0975, 0.0063]])
    Bc4 = np.array([[0.001, -0.0036, -0.0980, 0.0151]])
    # Bc = np.array([[0.,  -0., -0.0966,    0.0096, 0.0, -0.0,  -0.0971,    0.0097,     -0.0,   0.001,  -0.0975,    0.0063,     0.0001,     -0.0036,    -0.098, 0.0151  ]])
    
    B = np.stack([Bc1,Bc2,Bc3,Bc4], axis=2)
    hB = np.array([0.05, 0.10, 0.15, 0.20])

    C = np.stack([np.array([[0.9674]])],axis=2)
    hC = np.array([0.])

    Dc1 = np.array([[-963.3, -66.3, -1008.4, -6.5]])
    Dc2 = np.array([[1853.5, -228.3, 2776.6, -93.6]])
    Dc3 = np.array([[-2073.6, -327.3, -566.8, -164.7]])
    Dc4 = np.array([[-4101, -87.8, -2813, -53.6]])

    D = np.stack([Dc1,Dc2,Dc3,Dc4],axis=2)
    hD = np.array([0.05, 0.10, 0.15, 0.20])

    ddae = tdspy.DDAE(A=A, hA=hA, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)

    return ddae

    # controller = tdspy.controller.create_dynamic_controller(A,B,C,D)
    # return controller


def print_ddae(ddae: tds.DDAE):
    with np.printoptions(precision=4, linewidth=1000, suppress=True):
        print(f"E 2x2 matrix")
        print(ddae.E)
        print("-"*50)
        for i in range(ddae.mA):
            print(f"A[:,:,{i} - tau={ddae.hA[i]}")
            print(ddae.A[:,:,i])
            print("-"*50)
        for i in range(ddae.mB):
            print(f"B[:,:,{i} - tau={ddae.hB[i]}")
            print(ddae.B[:,:,i])
            print("-"*50)
        for i in range(ddae.mC):
            print(f"C[:,:,{i} - tau={ddae.hC[i]}")
            print(ddae.C[:,:,i])
            print("-"*50)
        for i in range(ddae.mD):
            print(f"D[:,:,{i} - tau={ddae.hD[i]}")
            print(ddae.D[:,:,i])
            print("-"*50)

if __name__ == "__main__":
    # Set up logging
    import logging
    logger = logging.getLogger("tdspy")
    logger.setLevel(logging.ERROR)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    rdde = generate_system()
    system_orig, BB, CC = tdspy.controller.interconnect3(rdde, [0,1,4,5], [0])
    
    system, BB, CC = tdspy.controller.interconnect3(rdde, [0,1,4,5], [0])
    
    cont = generate_controller_2()

    E, K, hK = concatenate_2x2_by_delays(cont.E, cont.A, cont.B, cont.C, cont.D, cont.hA, cont.hB, cont.hC, cont.hD)
    #K = np.random.rand(*K.shape)

    print_ddae(cont)
    print_ddae(tds.DDAE(K, hK, E))

    print_ddae(system)

    print(f"\n\n\n\n")
    # replace dynamics
    system._A = np.concatenate([system.A, np.stack([BB @ K[:,:,i] @ CC for i in range(K.shape[2])], axis=2)], axis=2)
    system._hA = np.r_[system.hA, hK]
    system.compress(inplace=True)

    # cr, cr_info = tds.roots(system,r=-10)

    # evolution = []
    # for i in range(100):
    #     system._A = np.concatenate([system_orig.A, np.stack([BB @ K[:,:,i] @ CC for i in range(K.shape[2])], axis=2)], axis=2)
    #     system._hA = np.r_[system_orig.hA, hK]
    #     system.compress(inplace=True)

    #     rstar, info = rightmost_root(system.E, system.A, system.hA, r=0)
    #     evolution.append(rstar)

    #     u = info.u[:, np.newaxis]
    #     v = info.v[:,np.newaxis]

    #     M = system.eval_char_matrix_derivative(rstar)
    #     coef = np.conj(u).T @ M @ v

    #     #gradient = np.real(coef * (np.conj(u).T @ BB).T @ (CC @ v).T)

    #     Kmask = np.full_like(K, fill_value=1, dtype=bool)
    #     matrix = (np.conj(u).T @ BB).T @ (CC @ v).T # (u* B).T (C v).T
    #     gradient = np.real((Kmask * hK) * matrix[:,:,np.newaxis])
    #     K = K - gradient
    
    # print(np.real(evolution))
    
    # print_ddae(tds.DDAE(K, hK, E))

    # import tdspy.plot
    # import matplotlib.pyplot as plt
    # plt.plot(evolution, "-o")
    # plt.show()

    cl = tds.ClosedLoop(rdde, 1, [0,1,4,5], [0], K0=K, hK=hK)

    print_ddae(cl)

    np.random.seed(10)
    E = cl.E
    P = cl._A
    hP = cl._hA
    K0 = np.random.rand(*cl.K.shape)
    hK = cl.hK
    B = cl.BB
    C = cl.CC

    Kmask = np.full_like(K0, fill_value=1, dtype=bool)
    Kshape = K0.shape

    from tdspy.stabopt.controller_bfgs import design_bfgs, func

    K = np.copy(K0)
    for i in range(10):
        alpha, jac = func(K.reshape(-1), E, P, hP, hK, Kshape, Kmask, B, C)
        #print(jac)
        #print("---------")
        print(alpha)
        K = K - 0.5*jac.reshape(K0.shape)

    #print(np.allclose(K0 - K0.reshape(-1).reshape(K0.shape), 0.0))
    
    #sol = design_bfgs(E, P, hP, K0, hK, B, C, options={"maxiter": 10, "disp": True, "gtol": 1e-5})
    #print(sol)

    A0 = np.concatenate(
        [
            P,
            np.stack([B @ K0[:,:,i] @ C for i in range(K0.shape[2])], axis=2)
        ],
        axis=2,
    )
    A = np.concatenate(
        [
            P,
            np.stack([B @ K[:,:,i] @ C for i in range(K.shape[2])], axis=2)
        ],
        axis=2,
    )
    hA = np.r_[hP, hK]


    import tdspy.plot
    import matplotlib.pyplot as plt

    cr0, _ = tds.roots(tds.DDAE(A0, hA, E), r=-10)
    cr, _ = tds.roots(tds.DDAE(A, hA, E), r=-10)
    

    plt.figure()
    print(cr0)
    print(cr)
    plt.scatter(np.real(cr0), np.imag(cr0), marker="x")
    plt.scatter(np.real(cr), np.imag(cr), marker="+")

    plt.show()
    
    
    

