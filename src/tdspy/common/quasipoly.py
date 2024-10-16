"""
TODO
"""

import numpy as np
import numpy.typing as npt

def qp_minimal_form(coefs: npt.NDArray, delays: npt.NDArray, atol: float=None, rtol: float=None):
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
    # TODO perform necessary checks

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

def neutral_from_qp(coefs, delays, ascending=True):
    """ Creates Neutral System from quasipolynomial

    TODO

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
    
    Returns: TODO
        tuple containing
            - A
            - hA
            - H
            - hH
    """

    if not ascending: # MATLAB like definition of s-powers coefficient
        coefs = coefs[:,::-1] # coefs of powers of s are in ascending order now

    # obtain minimal, sorted form (delays are in ascending order)
    coefs, delays = qp_minimal_form(coefs, delays)

    # obtain dimensions
    m = coefs.shape[0]
    n = coefs.shape[1] - 1

    # TODO make sure non zero dimensions

    if delays[0] != 0.0 or coefs[0,-1] == 0.0:
        raise ValueError("System can not be of advanced type!")
    # Normalize the coefficients
    coefs = coefs / coefs[0, -1] # TODO shouldn't it be highest non-zero and not -1

    if n < 1:
        # empty system TODO
        pass

    raise NotImplementedError(".")
