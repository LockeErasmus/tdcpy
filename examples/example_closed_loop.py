"""
Example 3.1 from TDS-CONTROL manal               
"""

import numpy as np
import tdspy as tds
import tdspy.plot

def generate_example01() -> tds.NDDE:
    """ generates example from TDS MATLAB manual (page 23)
    x'(t) = A0 x(t) + A1 x(t-\tau_1) + H1 \dot{x}(t-\tau_1) + H_2 \dot{x}(t-\tau_2)
    with 
    \tau_1 = 1, \tau_2 = 2 and
    A_0 = 1/4,  A_1 = 1/3
    H_1 = 3/4,  H_2 = -1/2."""


    # PLant parameters
    # Th = 14; Ta = 3; Td = 3; Tc = 25;

    # Kb = 0.24; Ka = 1; Kd = 0.94; Kc = 0.81; Ku = 0.39;
    # nh = 6.5; tb = 40; te = 13; td = 18;
    # tc = 2.8; nc = 9.2; u = 13.2;

    # dA = np.array([0,nh,tb,te,td,tc,nc]); dB = tu;

    A = np.stack([
        np.array([[1]]),
        np.array([[2]]),
    ], axis=2)

    hA = np.array([0, 1.])

    B = np.stack([
        np.array([[3]]),
    ], axis=2)

    hB = np.array([2])

    C = np.stack([
        np.array([[4]]),
    ], axis=2)

    hC = np.array([0])

    ddae = tdspy.DDAE(A=A, hA=hA,B=B, hB = hB, C = C, hC=hC)

    return ddae

def generate_example02() -> tds.DDAE:
    """ generates example from TDS MATLAB manual (help tds_create_ddae)
    6 x'(t) =  x(t) + 2 x(t-1) + 3 u(t-2)
    with 
    y(t)    = 4 x(t)   """

    E = np.array([[6]])

    A = np.stack([
        np.array([[1]]),
        np.array([[2]]),
    ], axis=2)

    hA = np.array([0, 1.])

    B = np.stack([
        np.array([[3]]),
    ], axis=2)

    hB = np.array([2])

    C = np.stack([
        np.array([[4]]),
    ], axis=2)

    hC = np.array([0])

    ddae = tdspy.DDAE(E=E,A=A, hA=hA,B=B, hB = hB, C = C, hC=hC)

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


    ddae = generate_example02()

    cr, cr0 = tdspy.roots(ddae, r=-0.7)

    import matplotlib.pyplot as plt
    tdspy.plot.eigen_plot(cr)
    plt.show()

