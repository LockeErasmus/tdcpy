"""
Set of functions for necessary quasipolynomial manipulation
"""

import numpy as np
import numpy.typing as npt

def sort_qp():
    """ I do not think this function is needed, but will implement it in future
    if necessary.
    """
    raise NotImplementedError(".")

def compress_qp(coefs: npt.NDArray, delays: npt.NDArray, atol: float=None, rtol: float=None) -> tuple[npt.NDArray, npt.NDArray]:
    """ Converts quasipolynomial to minimal form
    
    Minimal form of QP: No zero rows in coefs, last coef column is not zero
    column, delays are sorted in ascending order.

    Quasipolynomial is represented via matrix `coefs` of shape  (m, n+1), and
    vector of delays `delays` of shape (m), where the resulting quasipolynomial
    is defined as:

                 m-1                    n
        QP(s) =  SUM exp(-delays[i]*s) SUM coefs[i,j] * s**j
                 i=0                   j=0
    
    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        atol (float): absolute tolerance for determining if coefficient is
            sufficiently close to zero, default None, see numpy.isclose
        rtol (float): relative tolerance for determining if coefficient is
            sufficiently close to zero, default None, see numpy.isclose

    Returns:
        tuple containing
            - new_coefs (array): matrix definition of polynomial coefficients
            - new_delays (array): vector definition of associated delays
    """
    # TODO perform necessary checks ?
    new_delays = np.unique(delays) # sorted 1D array of unique delays
    m = new_delays.shape[0] # new number of delays
    n = coefs.shape[1] - 1
    new_coefs = np.zeros(shape=(m, n+1), dtype=coefs.dtype)
        
    for i in range(m):
        new_coefs[i, :] = np.sum(coefs[delays == new_delays[i], :], axis=0,
                                 keepdims=False)
    
    # prepare masks
    if atol or rtol:
        mask == np.isclose(new_coefs, 0, rtol=rtol, atol=atol)
    else:
        mask = new_coefs == 0    
    # obtain row and column masks    
    row_mask = ~mask.all(axis=1) # False if whole row of `new_coefs` == 0
    col_mask = ~mask.all(axis=0)
    # col_ix is last valid column, all columns with higher index are full of 0
    col_ix = col_mask.shape[0] - np.argmax(col_mask[::-1]) 
    col_mask[:] = True
    col_mask[col_ix:] = False # mark False all invalid
    
    new_coefs = new_coefs[np.ix_(row_mask, col_mask)]
    new_delays = new_delays[row_mask]
    return new_coefs, new_delays

def qp_to_ndde(coefs, delays, ascending=True) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """ Converts quasipolynomial into neutral delay differential equation

    Converts quasipolynomial defined via `coefs` and `delays`

                 m-1                    n
        QP(s) =  SUM exp(-delays[i]*s) SUM coefs[i,j] * s**j
                 i=0                   j=0

    into NDDE represented via arrays A, hA, H, hH

        dxdt(t) = A[:,:,0]*x(t - hA[0]) + ... + A[mA]*x(t - hA[mA])
            - H[:,:,0] * dxdt(t - hH[0]) - ... - H[:,:,mH] * dxdt(t - hH[mH])

    with mA number of delays associated with A and mH number of delays
    associated with H.

    Args:
        coefs (array): matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays (array): vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        ascending (bool): ordering of powers of s in each row, default ascending
            meaning that coefs[i,j] is associated to ith polynomial and jth
            power of s, setting this to False will default to original MATLAB
            behaviour, where coefs[i,j] is associated to ith polynomial and
            (n-j)th power of s
    
    Returns:
        tuple containing
            - A (array): matrices defining delay differential equation, with
                shape (n,n,mA)
            - hA (array): vector of delays associated with A
            - H (array): matrices defining delay difference equation, with
                shape (n,n,mH)
            - hH (array): vector of delays associated with H
    """
    if not ascending: # MATLAB like definition of s-powers coefficient
        coefs = coefs[:,::-1] # coefs of powers of s are in ascending order now

    # obtain minimal, sorted form (delays are in ascending order)
    coefs, delays = compress_qp(coefs, delays)

    # obtain dimensions
    m = coefs.shape[0] # number of delays
    n = coefs.shape[1] - 1 # degree/order

    if n < 1 or m < 1: # empty quasipolynomial -> just return empty system
        A, hA = np.zeros(shape=(0,0,0)), np.zeros(shape=(0,))
        H, hH = np.zeros(shape=(0,0,0)), np.zeros(shape=(0,))
        return A, hA, H, hA

    if delays[0] != 0.0 or coefs[0,-1] == 0.0:
        raise ValueError("System can not be of advanced type!")
    # Normalize the coefficients
    coefs = coefs / coefs[0, -1]

    # construct retarded part: A, hA
    hA = delays
    A = np.zeros(shape=(n,n,m), dtype=np.float64) # TODO dtype?
    np.fill_diagonal(A[:-1,1:,0], val=1.0) # inplace operration
    A[-1, :, :] = -coefs[:, :-1].T # highest power of s is omitted

    # check A[:,:,0] is zero matrix -> omit if True
    if np.all(A[:,:,0] == 0.):
        A = A[:,:,1:]
        hA = hA[1:]

    # construct neutral part: H, hH
    # please note, that if system is retarded (coefs[1:-1] is zero vector)
    #   array H has size 0 and shape (n,n,0) and hH has size 0 and shape (0,)
    H = np.zeros(shape=(n,n,0), dtype=A.dtype)
    hH = np.zeros(shape=(0,), dtype=hA.dtype)
    if np.any(coefs[1:, -1]): # returns False if all 0.0 or empty
        # non-empty and at least one non-zero coeficient -> neutral system
        hH = np.copy(delays)
        hH = np.zeros(shape=(n,n,m), dtype=np.float64) # TODO dtype?
        hH[-1, -1, :] = coefs[1:-1]
        # last step, filter out matrices which are zero -> in this case, simly
        mask = hH[-1, -1, :] == 0.0
        hH = hH[~mask]
        H = H[:,:,~mask]

    return A, hA, H, hH
