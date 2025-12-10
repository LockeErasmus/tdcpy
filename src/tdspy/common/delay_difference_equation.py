"""
Set of functions for obtaining and manipulation of delay difference equations
-----------------------------------------------------------------------------
Notes:
    1. these functions are internal, they do not operate via high level API
    2. these functions assume correct inputs, input types, etc. (that is to
        save computation time), use these function only if you know what you
        are doing
TODO:
    1. implemented also methods staring with `_` with no validation
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg

logger = logging.getLogger(__name__)


def ddae_to_diff(E, A, hA, uE=None, vE=None, **kwargs):
    """ Converts delay differential algebraic equation (DDAE) to delay
    difference equation

    DDAE dynamics represented by

    .. math::

        E \dot{x}(t) = A_0 x(t - h_{A,0}) + ... + A_{m_A} x(t - h_{A,m_A})
    into delay difference equation represented by
    
    .. math::
        D_0 x(t - h_{D,0}) + ... + D_{m_D} x(t - h_{D,m_D}) = 0

    Parameters:
    -----------
        E:  array
            left-side matrix of shape (n,n)
        A:  array
            right-side matrices of shape (n,n,m)
        hA: array
            vector of delays of shape (m,)
        uE: array, optional
            orthonormal basis for left null space of E, optional,
            default None means uE will be calculated via SVD
        vE: array, optional
            orthonormal basis for right null space of E, optional,
            default None means uE will be calculated via SVD
        **kwargs:
            tol: norm tolerance for considering matrix vanish, default 1e-14
            rcond (float): relative condition number. Singular values s smaller
                than rcond * max(s) are considered zero in null space
                construction, default 1e-12
            is_compressed: if True, hA is assumed to be in compressed form,
                i.e. hA[0] == 0, default True
        
    Returns:
    -----------
        tuple containing

            - D (array): right-side delay difference equation matrices
                of shape (p,p,q)
            - hD (array): right-side delay difference equation delays of
                shape (q,)

    Notes:
    ------
        1. n, m are assumed to be > 1
        2. A, hA is assumed to be in compressed form, i.e. hA[0] == 0
        3. if E is non-singular, D, hD are returned as empty arrays
        4. D, hD are returned in compressed form, i.e. no zero matrices in D
           (and associated delays in hD)
        5. if E is singular, the size of D, hD depends on the rank of E and
           the number of non-vanishing matrices uE.T @ A[i] @ vE
    
    Examples:
    --------
    >>> E = np.array([[1,0,0],[0,0,0],[0,0,1]])
    >>> A = np.zeros(shape=(3,3,3))
    >>> A[:,:,0] = np.array([[0,1,0],[0,0,0],[0,0,0]])
        >>> A[:,:,1] = np.array([[0,0,0],[1,0,0],[0,0,1]])
    >>> A[:,:,2] = np.array([[0,0,1],[0,0,0],[0,1,0]])
    >>> hA = np.array([0,1,2])
    >>> D,hD = ddae_to_diff(E,A,hA)
    >>> print("D=",D)
    >>> print("hD=",hD)
        D= [[[0. 0.]
          [0. 1.]
          [0. 0.]]
         [[0. 1.]
          [0. 0.]
          [0. 0.]]
         [[0. 0.]
          [0. 0.]
          [1. 0.]]]
        hD= [1 2]
    """
    rcond = kwargs.get("rcond", 1e-12)
    tol = kwargs.get("tol", 1e-14)
    is_compressed = kwargs.get("is_compressed", True)

    if uE is None:
        uE = linalg.null_space(E.T, rcond=rcond)
    if vE is None:
        vE = linalg.null_space(E, rcond=rcond) # TODO, SVD is now calculated twice
    
    # Case where E is non-singular -> return empty delay difference equation
    if uE.size == 0 or vE.size == 0:
        D = np.empty(shape=(A.shape[0], A.shape[1], 0), dtype=A.dtype)
        hD = np.empty(shape=(0,), dtype=hA.dtype)
        return D, hD

    # calculate norms for both nullspaces, select the bigger one
    norm_uE = linalg.norm(uE, ord=1, axis=None)
    norm_vE = linalg.norm(vE, ord=1, axis=None)
    norm_null = max(norm_uE, norm_vE)

    # calculate Di = uE.T @ Ai @ vE, D.shape == A.shape
    D = np.einsum(# more efficient way to obtain uE.T @ A[:,:,i] @ vE
        'ijk,jn->ink',
        np.einsum('ni,ijk->njk', uE.T, A),
        vE,
    )
    
    # select only Di =/= 0.0, i.e. Di sufficiently close to 0 are neglected
    mask = (linalg.norm(D, ord=1, axis=(0,1)) / norm_null) > tol
    D = D[:,:,mask]
    hD = hA[mask]

    return D, hD


def ndde_to_diff(H, hH, **kwargs):
    """ Converts NDDE to delay difference equation

    Parameters:
    -----------
        H:  array
            right-side matrices of shape (n,n,mH)
        hH: array
            vector of delays of shape (mH,)
        **kwargs:
            tol: norm tolerance for considering matrix vanish, default 1e-14

    Returns:
    -----------
        tuple containing    
            - D (array): right-side delay difference equation matrices
                of shape (n,n,mH+1)
            - hD (array): right-side delay difference equation delays of
                shape (mH+1,)
    
    Notes:
    ------

    For a NDDAE, the associated delay difference equation is given by
            
            I*x(t) + H[0]*x(t-hH[0]) + ... + H[mH]*x(t-hH[mH]) = 0          (1)

    1. n, mH are assumed to be > 0
    2. hH is assumed to be non-zero delays
    3. D, hD are returned in compressed form, i.e. no zero matrices in D
        (and associated delays in hD)
    4. D, hD are returned in normalized form, i.e. D[0] == I is omitted
    
    Examples:
    --------
    >>> H = np.zeros(shape=(2,2,2))
    >>> H[:,:,0] = np.array([[0,1],[0,0]])
    >>> H[:,:,1] = np.array([[0,0],[1,0]])
    >>> hH = np.array([1,2])
    >>> D,hD = ndde_to_diff(H,hH)
    >>> print("D=",D)
    >>> print("hD=",hD)
        D= [[[0. 0.]
          [0. 1.]]
         [[0. 1.]
          [0. 0.]]]
        hD= [1 2]

    """
    assert H.ndim == 3 and hH.ndim == 1
    assert H.shape[0] == H.shape[1] > 0 # square matrices
    assert H.shape[2] == hH.shape[0] > 0    
    assert np.all(hH != 0)

    n = H.shape[0]
    hD = np.concatenate([[0], hH], axis=0)
    D = np.concatenate([np.eye(n)[:,:,np.newaxis], H], axis=2)
    
    return H, hH


def _normalize_diff(D: npt.NDArray, hD: npt.NDArray) -> tuple:
    """ Normalizes delay difference equation 
    
    Transforms the delay difference equation such that the leading zero delay
    matrix D[0] equals identity (and can be omitted).

    Parameters:
    -----------
        D:  array
            right-side matrices of shape (n,n,m)
        hD: array
            vector of delays of shape (m,)

    Returns:
    -------
        tuple containing
            - D (array): 3D array representing matrices:
                [inv(D[0])*D[1], ... , inv(D[0])*D[m-1]]
            - hDD (array): array of non-zero delays of shape (m-1,)

    Notes:
        1. D[0] is assumed to be invertible
        2. hD[0] == 0

    Examples:
    ---------
    >>> D = np.zeros(shape=(2,2,3))
    >>> D[:,:,0] = np.array([[1,0],[0,1]])
    >>> D[:,:,1] = np.array([[0,1],[1,0]])
    >>> D[:,:,2] = np.array([[1,1],[0,0]])
    >>> hD = np.array([0,1,2])
    >>> DD,hDD = _normalize_diff(D,hD)
    >>> print("DD=",DD)
    >>> print("hDD=",hDD)
        DD= [[[0. 1.]
          [1. 0.]]
         [[1. 1.]
          [0. 0.]]]
        hDD= [1 2]
    
    """

    assert D.ndim == 3 and hD.ndim == 1
    assert D.shape[0] == D.shape[1] > 0
    assert D.shape[2] == hD.shape[0] >= 1
    assert hD[0] == 0

    hDD = hD[1:]
    n, m = D.shape[0], hDD.shape[0]
    DD = np.zeros(shape=(n, n, m))

    if m == 1:
        DD[:,:,0], res, rank, eig = linalg.lstsq(D[:,:,0], D[:,:,1])
    else:
        # LU decomposition for having inverse of d[:,:,0]
        P, L, U = linalg.lu(D[:,:,0])
        for i in range(1, m+1):
            # Di = inv(A0) @ Ai
            LPD, _, _, _ = linalg.lstsq(L, P @ D[:,:,i])
            DD[:,:, i-1], _, _, _ = linalg.lstsq(U, LPD)
        
    return DD, hDD # return normalized


def normalize_diff(D: npt.NDArray, hD: npt.NDArray) -> tuple:
    """ Normalizes delay difference equation

    Transforms the delay difference equation such that the leading zero delay
    matrix D[0] equals identity (and can be omitted).

    Parameters:
    -----------
        D:   array
            right-side matrices of shape (n,n,m)
        hD (array): vector of delays of shape (m,)

    Returns:
    -------
        tuple containing
        
            - D (array): 3D array representing matrices:
                [inv(D[0])*D[1], ... , inv(D[0])*D[m-1]]
            - hDD (array): array of non-zero delays
    
    Notes:
    ------
        1. m > 2 is assumed
        2. D[0] is assumed to be invertible
        3. hD[0] == 0

    Examples:
    ---------
    >>> D = np.zeros(shape=(2,2,3))
    >>> D[:,:,0] = np.array([[1,0],[0,1]])
    >>> D[:,:,1] = np.array([[0,1],[1,0]])
    >>> D[:,:,2] = np.array([[1,1],[0,0]])
    >>> hD = np.array([0,1,2])
    >>> DD,hDD = normalize_diff(D,hD)
    >>> print("DD=",DD)
    >>> print("hDD=",hDD)
        DD= [[[0. 1.]
          [1. 0.]]
         [[1. 1.]
          [0. 0.]]]
        hDD= [1 2]

    """
    assert D.ndim == 3 and hD.ndim == 1
    assert D.shape[0] == D.shape[1] > 0
    assert D.shape[2] == hD.shape[0] > 1
    assert hD[0] == 0
    # TODO, check if D[:,:,0] is invertible by SVD?

    DD, hDD = _normalize_diff(D, hD)

    return DD, hDD

    

    