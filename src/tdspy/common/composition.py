"""
Set of functions for TDS composition
"""

import numpy as np
import numpy.typing as npt

def concatenate_2x2_by_delays(E, A, B, C, D, hA, hB, hC, hD, EE=None, AA=None, hAA=None):
    """ Concatenates system into compact form respecting delay vectors

    Assumes system is defined as

        E dxdt = A[:,:,0]*x(t-hA[0]) + ... + A[:,:,n] x(t-hA[n]) +
                 + B[:,:,0]*u(t-hB[0]) + ... + B[:,:,m] u(t-hB[n])
        
            y  = C[:,:,0]*x(t-hC[0]) + ... + C[:,:,p] x(t-hC[p]) +
                 + D[:,:,0]*u(t-hD[0]) + ... + D[:,:,q] u(t-hD[q])
    
    Concatenates the system into:

        E*dx1dt = A*[:,:,0]*x2(t-hA*[0]) + ... + A*[:,:,n*] x2(t-hA*[n*])
    
    where:
        x1 := [x^T y^T]^T
        x2 := [x^T u^T]^T
    and therefore:
        hA* = [hA, hB, hC, hD]
        n* = n+m+p+q
    left hand-side matrix:
        E* = [E, 0]
             [0, 0]
    right hand-side array:
        A*[:,:,:n] = [A, 0]  
                     [0, 0]
        A*[:,:,n:n+m] = [0, B]  
                        [0, 0]
        A*[:,:,n+m:n+m+p] = [0, 0]  
                            [C, 0]
        A*[:,:,n+m+p:] = [0, 0]
                         [0, D]

    Args:
        TODO
    
    Returns:
        tuple containing:

            - EE (array): 2d array of concatenated LHS
            - AA (array): 3d array of concatenated RHS
            - hAA (array): 1d vector of concatenated delays associated with RHS
    """
    # TODO perform necessary checks

    # non emtpy assumption

    # if necessary, create empty
    nrows = A.shape[0] + C.shape[0]
    ncols = A.shape[1] + B.shape[1]
    n, m, p, q = hA.shape[0], hB.shape[0], hC.shape[0], hD.shape[0]

    # hAA - new delay vector
    if hAA is None:
        hAA = np.concatenate([hA, hB, hC, hD], axis=0)
    else:
        hAA[:n] = hA
        hAA[n:n+m] = hB
        hAA[n+m:n+m+p] = hC
        hAA[n+m+p:] = hD

    # EE - LHS matrix
    if EE is None:
        EE = np.zeros(shape=(nrows, ncols), dtype=E.dtype)
    EE[:E.shape[0], :E.shape[1]] = E

    # AA - RHS array
    if AA is None:
        AA = np.zeros(shape=(nrows, ncols, hAA.shape[0]), dtype=A.dtype)

    AA[:A.shape[0],:A.shape[1],:n] = A
    AA[:A.shape[0],A.shape[1]:,n:n+m] = B
    AA[A.shape[0]:,:C.shape[1],n+m:n+m+p] = C
    AA[A.shape[0]:,C.shape[1]:,n+m+p:] = D

    return EE, AA, hAA

def interconnect():
    pass