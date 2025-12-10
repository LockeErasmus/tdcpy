## Testing the function tds.cd
## Check with the example ex2_neutral from Ch3_DesigningStabilizingControllers

import numpy as np
import matplotlib.pyplot as plt
import tdspy as tds
import tdspy.controller
from numpy.linalg import inv
from tdspy.common.composition import concatenate_2x2_by_delays
from tdspy.stability.gamma_r import gamma_normalized_diff, gamma_diff
from tdspy.common.delay_difference_equation import ddae_to_diff, normalize_diff


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
    K, hK = np.zeros(shape=(1,3,1)), np.zeros(shape=(1,))       
    K[:,:,0], hK[0] = K1[:,:,1], hK1[1]
    
    # forming the closed-loop
    cl = tds.ClosedLoop(ddae, 0, [0,1,2], [0], K, hK=hK)
    print_ddae(cl)

    cl_ddae = tds.DDAE(E=cl.E,A=cl.A,hA=cl.hA)  # extract the ddae of the closed-loop
    cl_dde = cl_ddae.get_delay_difference_equation()
    D, hD = ddae_to_diff(E=cl.E, A=cl.A, hA=cl.hA, uE=cl.uE, vE=cl.vE)
    DD, hDD = normalize_diff(D, hD)
    print_ddae(cl_ddae)

    #-----------------TESTING IT FOR K = controller parameters-----------------------#
    # Testing whether spectral abscissa of the ddae and cl match the spectral abscissa from tds-control
    # K = controller parameters from Pieter's optimization
    # OK
    print(f"SA of plant: {tds.spectral_abscissa(ddae, r=-0.1)}")                # must be = 0.1081
    
    # NOT OK
    cl_roots, rootsInfo = tds.roots(cl_ddae,r=-0.1)
    print(f"Roots of closed-loop: {np.max(np.real(cl_roots))}")                # must be = -0.0309 
    print(f"Roots of closed-loop: {tds.spectral_abscissa(cl_ddae, r=-0.1)}")    # must be = -0.0309
    cd,cdInfo = tds.spectral_abscissa_diff(cl_dde)                             # must be = -0.0308894, more or less correct
    print(f"CD of diff: {cd}")
    gamma, gammaInfo = tds.gamma(cl_ddae,r=0,is_compressed=0)   
    print(f"gamma0 of diff: {gamma}")
    # gamma_norm_diff, gamma_norm_diff_Info = gamma_normalized_diff(cl_dde.A[:,:,1:], cl_dde.hA[1:], r=0, is_compressed=1)  
    g, info = gamma_normalized_diff(DD, hDD, 0, correction=True,is_compressed=0) 
    print(f"gamma_norm_diff of diff: {g}")

    #-----------------TESTING IT FOR K = 0.01+np.zeros(shape=(1,3,1))----------------#

    K = 0.01+np.zeros_like(K)
    cl2 = tds.ClosedLoop(ddae, 0, [0,1,2], [0], K, hK=hK)
    print_ddae(cl2)

    cl2_ddae = tds.DDAE(E=cl2.E,A=cl2.A,hA=cl2.hA)  # extract the ddae of the closed-loop
    cl2_dde = cl2_ddae.get_delay_difference_equation()
    D, hD = ddae_to_diff(E=cl2.E, A=cl2.A, hA=cl2.hA, uE=cl2.uE, vE=cl2.vE)
    DD, hDD = normalize_diff(D, hD)
    print_ddae(cl2_ddae)

    # Testing whether spectral abscissa of the ddae and cl match the spectral abscissa from tds-control
    # OK
    print(f"SA of plant: {tds.spectral_abscissa(ddae, r=-0.1)}")                # must be = 0.1081
    cl2_roots, rootsInfo = tds.roots(cl2_ddae,r=-0.1)
    print(f"Roots of closed-loop: {np.max(np.real(cl2_roots))}")                    # must be = 0.1067
    print(f"Roots of closed-loop: {tds.spectral_abscissa(cl2_ddae, r=-0.1)}")       # must be = 0.1067

    # NOT OK
    g2, info2 = gamma_normalized_diff(DD, hDD, 0, correction=True,is_compressed=0) 
    gamma2, gammaInfo2 = tds.gamma(cl2_ddae,r=0,is_compressed=False)                # must be
    cd2,cdInfo = tds.spectral_abscissa_diff(cl2_dde)                             # must be = -0.8657, ours gives -inf
    print(f"CD of diff: {cd2}")