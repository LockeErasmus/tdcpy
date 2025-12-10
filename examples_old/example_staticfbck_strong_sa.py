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

import tdspy.controller
import tdspy.plot
from tdspy.common.delay_difference_equation import ddae_to_diff
from tdspy.common.composition import concatenate_2x2_by_delays
from tdspy.stability.characteristic_roots import rightmost_root, RightmostRootInfo


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
    Example 3.3 taken from TDS-Manual
    x'(t)   =   A x(t)  +   B u(t-tau1)
    y(t)    =   C x(t)  +   D1 u(t-tau2)    +   D2 u(t-tau3)
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
    B = np.stack([Bu],axis=2)
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
 
def generate_controller() -> tds.DDAE:
    """ generates static output feedback controller according to the paper
    u   =   Dc * y(t)
    """
    controller = tdspy.controller.create_static_controller(
        K = np.array([[     0.0409,     0.0612,     0.3837  ]])
    )

    return controller


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
    diff = ddae.get_delay_difference_equation()
    cont = generate_controller()
    print_ddae(ddae)
    E, K, hK = concatenate_2x2_by_delays(cont.E, cont.A, cont.B, cont.C, cont.D, cont.hA, cont.hB, cont.hC, cont.hD)
    
    plant_sa = tds.spectral_abscissa(ddae, r=-10)
    print(plant_sa)

    cr, _ = tds.roots(ddae)

    print(cr)

    # cl = tdspy.ClosedLoop(ddae, 0, [0,1,2], [0], K0=K, hK=hK)
    # print_ddae(cl)
    # K = np.array()

    # print(f"The DDAE is essentially neutral={ddae.is_essentially_neutral}")

    # cd = tds.cd(ddae)

    # print(f"strogn spectral abscissa of associated DIFF {cd=}

    # cl = tds.ClosedLoop(ddae, 0, [0,1,2], [0], K0=K, hK=hK)
    cl = tds.ClosedLoop(ddae, 0, [0,1,2], [0], K0=np.zeros(K.shape), hK=hK)
    print_ddae(cl)

    np.random.seed(10)
    
    E = cl.E
    P = cl._A
    hP = cl._hA
    K0 = np.random.rand(*cl.K.shape)
    K0 = np.ones(cl.K.shape)
    hK = cl.hK
    B = cl.BB
    C = cl.CC

    Kmask = np.full_like(K0, fill_value=1, dtype=bool)
    Kshape = K0.shape

    from tdspy.stabopt.controller_bfgs import design_bfgs, stab_opt
    from tdspy.stabopt.gradients import func_sa, func_cd, gradient_test


    # check if cl contains a delay-difference 
    # get delay difference equation from cl

    K_ddae = cl.controller

    cl_ddae = tds.DDAE(E=cl.E,A=cl.A,hA=cl.hA)

    # cl_ddae = tds.DDAE(E=cl.E,A=cl.A,hA=cl.hA,B=cl.BB[:,:,np.newaxis],hB=np.array([0]),C=cl.CC[:,:,np.newaxis],hC=np.array([0]))
    print_ddae(cl_ddae)
    
    cl_dde = cl_ddae.get_delay_difference_equation()
    print_ddae(cl_ddae)
    cl_dde.A
    cl_dde.hA
    plant_sa = tds.spectral_abscissa(ddae, r=-0.1)
    cl_sa = tds.spectral_abscissa(cl_ddae, r=-0.1)

    print(f"SA of plant: {plant_sa}")
    print(f"SA of CL: {cl_sa}")


    # case 1: dde 
    if cl_dde.A.shape[2] == 1:  # system is retarded
        func = func_sa          
    else:                       # system is neutral
        func = func_cd          




    # test for DIFF dependency
    from tdspy.stabopt.utils import diff_dependency_mask

    g_numerical, g_analytical = gradient_test(func_cd, x=np.random.rand(K0.size), args=(E, P, hP, hK, Kmask, B, C))

    sol = design_bfgs(E, P, hP, K0, hK, B, C, options={"disp": True, "eps":0.1})

    # here, user specifies adjustability of controller parameters
    Kmask = np.full_like(K0, fill_value=True, dtype=bool) # all parameters adjustable

    r = diff_dependency_mask(Kmask, cl.uE, cl.vE, cl.BB, cl.CC)

    print(r.shape)
    for i in range(r.shape[2]):
        print("Parameter mask Kmask[:,:,{i}]:")
        print(Kmask[:,:,i])
        print(f"DIFF_MASK[:,:,{i}] - tau={cl.hK[i]}")
        print(r[:,:,i])
        print("-"*50)
    
    if np.all(~r):
        print("DIFF IS INDEPENDENT OF CONTROLLER PARAMETERS")
    else:
        print("DIFF IS DEPENDENT")
    
    sol = stab_opt(E, P, hP, K0, hK, B, C, options={"disp": True, "eps":0.1})
    K = sol.x.reshape(K.shape)