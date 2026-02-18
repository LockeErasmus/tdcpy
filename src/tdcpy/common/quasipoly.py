# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

"""
Set of functions for necessary quasipolynomial manipulation
-----------------------------------------------------------

Implemented functions:

1. `compress_qp`: converts a quasipolynomial to minimal form
2. `qp_to_ndde`: returns an ndde represented by `(H,hH,A,hA)` given a quasipolynomial of the representation (coeffs,delays)

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

    ..  math::
    
        QP(s) =  \\sum\\limits_{i=0}^{m-1} exp(-delays[i]*s) \\sum\\limits_{j=0}^{n} coefs[i,j] * s^j
    
    Parameters
    ----------
    coeffs: array
        matrix definition of polynomial coefficients (each row
        represents polynomial coefficients corresponding to delay)
    delays: array
        vector definition of associated delays (each delay
        corresponds to row in `coefs`)
    atol: float, optional
        absolute tolerance for determining if coefficient is
        sufficiently close to zero, default None, see numpy.isclose
    rtol: float, optional
        relative tolerance for determining if coefficient is
        sufficiently close to zero, default None, see numpy.isclose
                 
    Returns
    -------
    tuple
        A tuple containing

        new_coefs : array
            matrix definition of polynomial coefficients
        new_delays : array
            vector definition of associated delays
  
    Notes
    -----
    1. if all coefficients are close to zero, returns empty arrays
    2. if coefs is empty, returns coefs and delays unchanged
    3. if delays is empty, returns coefs and delays unchanged
    4. if coefs and delays have inconsistent shapes, raises ValueError
    5. if rtol or atol are negative, raises ValueError
    6. if coefs is not 2D array, raises ValueError
    7. if delays is not 1D array, raises ValueError
    8. if coefs.shape[0] != delays.shape[0], raises ValueError

    Examples
    --------
    >>> import numpy as np
    >>> from tdcpy.common.quasipoly import compress_qp
    >>> coefs = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 0.0]])
    >>> delays = np.array([0.0, 1.0, 2.0])
    >>> new_coefs, new_delays = compress_qp(coefs, delays)
    >>> new_coefs
    array([[0., 1.],
           [0., 2.]])
    >>> new_delays
    array([0., 1.])
    >>> coefs = np.array([[0.0, 0.0], [0.0, 0.0]])
    >>> delays = np.array([0.0, 1.0])
    >>> new_coefs, new_delays = compress_qp(coefs, delays)
    >>> new_coefs
    array([], shape=(0, 2), dtype=float64)
    >>> new_delays
    array([], dtype=float64)
        
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

    ..  math::

        QP(s) =  \\sum\\limits_{i=0}^{m-1} exp(-delays[i]*s) \\sum\\limits_{j=0}^{n} coefs[i,j] * s^j

    into NDDE represented via arrays A, hA, H, hH

    .. math::

        \dot{x}(t) = A_0*x(t - hA_0) + ... + A_{mA}*x(t - hA_{mA})
            - H_0* \dot{x}(t - hH_0) - ... - H_{mH}* \dot{x}(t - hH_{mH})

    with mA number of delays associated with A and mH number of delays
    associated with H.

    Parameters
    ----------
        coeffs : array
            matrix definition of polynomial coefficients (each row
            represents polynomial coefficients corresponding to delay)
        delays : array
            vector definition of associated delays (each delay
            corresponds to row in `coefs`)
        ascending : bool, optional
            ordering of powers of s in each row, default ascending meaning that coefs[i,j]
            is associated to ith polynomial and jth power of s, setting this to False will default to original MATLAB
            behaviour, where coefs[i,j] is associated to ith polynomial and (n-j)th power of s
    
    Returns
    -------
    tuple
        A tuple containing

        A : array
            matrices defining delay differential equation, with
            shape (n,n,mA)
        hA : array
            vector of delays associated with A
        H : array
            matrices defining delay difference equation, with
            shape (n,n,mH)
        hH : array
            vector of delays associated with H

    Notes
    -----
    1. if all coefficients are close to zero, returns empty arrays
    2. if coefs is empty, returns empty arrays
    3. if delays is empty, returns empty arrays
    4. if coefs and delays have inconsistent shapes, raises ValueError
    5. if system is of advanced type (delay[0] != 0.0 or coefs[0,-1] == 0.0), raises ValueError
    
    Examples
    --------
    >>> import numpy as np
    >>> from tdcpy.common.quasipoly import qp_to_ndde
    >>> coefs = np.array([[1.0, 0.0], [0.0, 2.0]])
    >>> delays = np.array([0.0, 1.0])
    >>> A, hA, H, hH = qp_to_ndde(coefs, delays, ascending=False)
    >>> A
    array([[[-2.]]])
    >>> hA
    array([1.])
    >>> H
    array([], shape=(1, 1, 0), dtype=float64)
    >>> hH
    array([], dtype=float64)
    >>> coefs = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    >>> delays = np.array([0.0, 1.0, 2.0])
    >>> A, hA, H, hH = qp_to_ndde(coefs, delays)
    >>> A
    array([[[-0., -0.]]])
    >>> A.shape
    (1, 1, 2)
    >>> hA
    array([1., 2.])
    >>> H
    array([[[2., 3.]]])
    >>> H.shape
    (1, 1, 2)
    >>> hH
    array([1., 2.])

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
        H = np.zeros(shape=(n,n,m), dtype=np.float64) # TODO dtype?
        H[-1, -1, 1:] = coefs[1:,-1]
        # last step, filter out matrices which are zero -> in this case, simply
        mask = H[-1, -1, :] == 0.0
        hH = hH[~mask]
        H = H[:,:,~mask]

    return A, hA, H, hH

if __name__ == "__main__":
    import doctest
    doctest.testmod()