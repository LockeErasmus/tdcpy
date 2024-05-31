"""

"""
import numpy as np
import matplotlib.pyplot as plt
import tdspy as tds

def crate_system(m1=1.1, m2=0.514, k1=1768, k2=424, c1=4.43, c2=2.41) -> tuple:
    """ returns A, Bu, Bd with correct shapes
    
    default values taken from:
        [1] Yuksel, Can Kutlu, Silviu-Iulian Niculescu, and Tomáš Vyhlídal.
        "A Spectrum-based Filter Design for Periodic Output Regulation of 
        Systems with Dead-Time." IFAC-PapersOnLine 56.2 (2023): 917-922.
    """
    A = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [-(k1+k2)/m1, k2/m1, -(c1+c2)/m1, c2/m1],
        [k1/m2, -k2/m2, c1/m2, -c2/m2],
    ])
    Bu = np.array([[0],[0],[1/m1],[0]])
    Bd = np.array([[0],[0],[-1/m1],[1/m2]])
    C = np.array([[0, 1, 0, 0]])
    return A, Bu, Bd, C

def create_cl_ddae(kp=-10.0, tau1=0.2, tau2=0.3) -> tds.DDAE:
    """ TODO docstring """
    A, Bu, Bd, C = crate_system() # create system with defaults
    ns = 4 # number of states in primary system
    n = 7 # shape of E, Ai -> (n,n)
    E = np.zeros(shape=(n,n), dtype=np.float64)
    E[:ns, :ns] = np.eye(ns)
    
    # construct matrix A0
    A0 = np.zeros(shape=(n,n), dtype=np.float64)
    A0[:ns, :ns] = A
    A0[[ns], :ns] = C
    A0[[ns, ns+1], [ns, ns+2]] = -1
    A0[[ns+1, ns+2], [ns+1, ns]] = 1

    # construct matrix A1
    A1 = np.zeros(shape=(n,n), dtype=np.float64)
    A1[:ns, [ns+1]] = Bu

    # construct matrix A2
    A2 = np.zeros(shape=(n,n), dtype=np.float64)
    A1[ns+1, ns+1] = -1

    # construct input matrix
    # TODO - not necessary for stability

    return tds.DDAE(E=E, A=np.stack([A0, A1, A2], axis=2), hA=np.array([0, tau1, tau2]))


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

