## Testing the gradient_cd function
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

def generate_controller1() -> tds.DDAE:
    """ generates static output feedback controller according to the paper
    u   =   Dc * y(t)
    """
    controller = tds.controller.create_static_controller(
        K = np.array([[     0.0409,     0.0612,     0.3837  ]])
    )

    return controller

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
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create ddae, controller
    ddae = create_system()
    diff = ddae.get_delay_difference_equation()
    cont = generate_controller1()
    print_ddae(ddae)
    E, K1, hK1 = concatenate_2x2_by_delays(cont.E, cont.A, cont.B, cont.C, cont.D, cont.hA, cont.hB, cont.hC, cont.hD)
    
    # workaround to get a single delay term for the controller
    K, hK = 0.01+np.zeros(shape=(1,3,1)), np.zeros(shape=(1,))       
    x = K[:,:,0].reshape(-1)
   
    # forming the closed-loop
    cl = tds.ClosedLoop(ddae, 0, [0,1,2], [0], K, hK=hK)
    print_ddae(cl)



    E = cl.E
    P = cl._A
    hP = cl._hA
    K0 = 0.01+np.array(np.zeros(shape=(1,3,1)))
    hK = cl.hK
    B = cl.BB
    C = cl.CC
    uE = cl.uE
    vE = cl.vE

    Kmask = np.full_like(K, fill_value=1, dtype=bool)
    Kshape = K0

    
    cl_ddae = tds.DDAE(E=cl.E,A=cl.A,hA=cl.hA)  # extract the ddae of the closed-loop
    cl_dde = cl_ddae.get_delay_difference_equation()
    D, hD = ddae_to_diff(E=cl.E, A=cl.A, hA=cl.hA, uE=cl.uE, vE=cl.vE)
    DD, hDD = normalize_diff(D, hD)

    ### TEST: Check if spectral abscissa is correct, if YES, continue
    cr, cr_info = tds.roots(cl_ddae, r=-0.1)            # should be = 0.1067

    print_ddae(cl_ddae)

    ### TEST: Check if the gamma0 and cd match the expected values, if YES, continue
    gamma0, out = gamma_normalized_diff(DD, hDD, r=0, correction=True, is_compressed=0)                 # = 0.2898, must be = 0.2898

    sa_diff, cdInfo = spectral_abscissa_diff(DD,hDD,r=-0.1)                         # = -0.8657, ok
    # NOT OK
    print(f"out.th is: {out.th}")                                           # must be = [0. 3.1416 5.6549]
    print(f"out.s is: {out.s}")                                             # must be = -0.2756+ 0.0896j (if we mupltiply by -j, then it's correct)
    print(f"out.u is: {out.u}")                                             # incorrect values
    print(f"out.v is: {out.v}")                                             # incorrect values                                     # incorrect values


    # # creating permutation matrix p1
    # p1 = np.array(np.eye(4,4,1))

    # p1[0,0],p1[1,0] = 0,1
    # p1[0,1],p1[1,1] = 1,0
    # p1[3,2],p1[1,2] = 1,0

    # # permutated matrices
    # DD1 = np.zeros_like(DD)
    # hDD1 = hDD
    # DD1 = np.einsum('ij,jkl->ikl',p1, DD)

    # gamma0, out = gamma_normalized_diff(D1[:,:,1:], hD1[1:], r=0, is_compressed=0)         



    # cl_dde.A = p1@cl_dde.A

    # # setting parameters acc to tds-control
    # cd = -0.8657
    # th = np.array([0.,3.1416, 1.8850])

    # from collections import namedtuple
    # GammaInfo = namedtuple("GammaInfo", ["th", "M", "s", "u", "v"])
    # out = GammaInfo(np.r_[0, th_star], M_star, eig_star, u_star, v_star)
    cd = sa_diff

    g_numerical, g_analytical = gradient_test(func=func_cd, x=K.reshape(-1), args=(E, P, hP, Kmask, hK, B, C, uE, vE, out, cd))

    DD