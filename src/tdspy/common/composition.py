"""
Set of functions for TDS composition
"""

import numpy as np
import numpy.typing as npt

def concatenate_2x2_by_delays(E: npt.NDArray, A: npt.NDArray, B: npt.NDArray,
                              C: npt.NDArray, D: npt.NDArray, hA: npt.NDArray,
                              hB: npt.NDArray, hC: npt.NDArray, hD: npt.NDArray,
                              EE: npt.NDArray=None, AA: npt.NDArray=None,
                              hAA: npt.NDArray=None):
    """ Concatenates system into compact form respecting delay vectors

    Parameters:
    -----------
        E:  array
            RHS matrix
        A:  array
            left hand side matrices of DDAE, assumed non-empty
        B:  array
            3D array of representing input matrices
        C:  array
            3D array of representing output matrices
        D:  array
            left hand side matrices of DDAE, assumed non-empty
        hA: array
            vector of delays associated with array A
        hB: array
            vector of delays associated with array B
        hC: array
            vector of delays associated with array C
        hD: array
            vector of delays associated with array D
        EE: array, optional
            if provided, used as LHS matrix of the concatenated system
        AA: array, optional
            if provided, used as RHS 3D array of the concatenated system
        hAA: array, optional
            if provided, used as delay vector of the concatenated system

    Returns:
    --------
        tuple containing:

            - EE (array): 2d array of concatenated LHS
            - AA (array): 3d array of concatenated RHS
            - hAA (array): 1d vector of concatenated delays associated with RHS

    Notes:
    ------
    Assumes system is defined as

    .. math::

        E \dot{x}(t) = A_0 x(t - h_{A,0}) + ... + A_{n} x(t - h_{A,n}) + 
                    + B_0 u(t - h_{B,0}) + ... + B_{m} u(t - h_{B,m})
            
                y(t)  = C_0 x(t - h_{C,0}) + ... + C_{p} x(t - h_{C,p}) + 
                    + D_0 u(t - h_{D,0}) + ... + D_{q} u(t - h_{D,q})

    Concatenates the system into:
    .. math::

        E \dot{x1}(t) = A^*_0 x2(t - hA^*_0) + ... + A^*_{n^*} x2(t - hA^*_{n^*})
    
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
                         
    - Assumes all input arrays are non-empty
    - If EE, AA, hAA are provided, they are updated in place and returned
    - If EE, AA, hAA are not provided, they are created and returned
    - The resulting system is in the form suitable for creating ClosedLoop object

    Examples:
    ---------
    >>> import numpy as np
    >>> from tdspy.common.composition import concatenate_2x2_by_delays
    >>> E = np.array([[1, 0], [0, 0]])
    >>> A = np.array([[[0, -1], [1, 0]], [[0, 0], [0, 0]]])
    >>> B = np.array([[[0], [1]], [[0], [0]]])
    >>> C = np.array([[[1, 0]], [[0, 0]]])
    >>> D = np.array([[[0]], [[1]]])
    >>> hA = np.array([0., 1.])
    >>> hB = np.array([0.])
    >>> hC = np.array([0.,0.])
    >>> hD = np.array([0.])
    >>> EE, AA, hAA = concatenate_2x2_by_delays(E, A, B, C, D, hA, hB, hC, hD)
    >>> EE
    array([[1, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    >>> AA[:,:,0]
    array([[0, 1, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    >>> AA[:,:,1]
    array([[-1,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  0,  0]])
    >>> AA[:,:,2]
    array([[0, 0, 0, 1],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])
    >>> hAA
    array([0., 1., 0., 0., 0., 0.])

    """
    # TODO perform necessary checks
    assert A.size > 0 and B.size > 0 and C.size > 0 and D.size > 0, "Only non-empty arrays are supported!"
    assert hA.size > 0 and hB.size > 0 and hC.size > 0 and hD.size > 0, "Only non-empty delay vectors are supported!"
    assert E.size > 0, "Only non-empty arrays are supported!"
    assert A.ndim == 3 and B.ndim == 3 and C.ndim == 3 and D.ndim == 3, "A, B, C, D must be 3D arrays!"
    assert E.ndim == 2, "E must be a 2D array!"
    assert hA.ndim == 1 and hB.ndim == 1 and hC.ndim == 1 and hD.ndim == 1, "Delay vectors must be 1D!"
    assert A.shape[2] == hA.shape[0], "Inconsistent shape between A and hA!"
    assert B.shape[2] == hB.shape[0], "Inconsistent shape between B and hB!"
    assert C.shape[2] == hC.shape[0], "Inconsistent shape between C and hC!"
    assert D.shape[2] == hD.shape[0], "Inconsistent shape between D and hD!"

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

    # AA - RHS 3D array
    if AA is None:
        AA = np.zeros(shape=(nrows, ncols, hAA.shape[0]), dtype=A.dtype)

    AA[:A.shape[0],:A.shape[1],:n] = A
    AA[:A.shape[0],A.shape[1]:,n:n+m] = B
    AA[A.shape[0]:,:C.shape[1],n+m:n+m+p] = C
    AA[A.shape[0]:,C.shape[1]:,n+m+p:] = D

    return EE, AA, hAA

def interconnect():
    pass


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import doctest
    doctest.testmod()