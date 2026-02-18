# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

"""
Set of function for representation compressions
-----------------------------------------------

The `compress` module consists of functions for obtaining a minimal representaation of a 
given set of matrices and delays.

compression := obtaining minimal sorted representation of "something"

Implemented functions:

1. `compress_matrices_delays`: creates a compressed matrix representation ``(A*, hA*)`` by removing duplicate delays
2. `compress_bool_matrices_delays`: removes delay duplicates and sorts matrices in ascending of delays
3. `sort_matrices_delays`: sorts the matrices in ascending order of the delays

TODO:
    ---

Notes:
    ---
"""

import numpy as np
import numpy.typing as npt

def compress_matrices_delays(A: npt.NDArray, hA: npt.NDArray, rtol=1e-5, atol=1e-8):
    """ Compresses matrices - delays representation
    
    Removes delay duplicates, sorts delays into ascending order and removes
    matrices close to zero, i.e. converts the representation (A, hA):

    .. math::
        A_0 x(t - h_{A,0}) + ... + A_{mA} x(t - h_{A,mA})                           
    
    into representation (A*, hA*), where:

    1. matrices A*[i] are NOT close to zero
    2. hA* does not contain duplicates

    Parameters
    ----------
    A : array
        3D array of stacked matrices (axis 2)
    hA : array
        1D array (vector) of delays
    rtol : float
        relative tolerance for determining matrix element is
        zero, default 1e-5
    atol : float
        absolute tolerance for determining matrix element is 
        zero, default 1e-8
    
    Returns
    -------
    tuple
        A tuple containing

        compressed_A : array
            compressed representation of A
        compressed_hA : array
            compressed vector of delays

    Notes
    -----
    1. the sort is "stable" (see numpy.argsort implementation)
    2. if all matrices are close to zero, returns empty array with
        shape (A.shape[0], A.shape[1], 0) and empty hA   array
    3. if A is empty, returns A and hA unchanged
    4. if hA is empty, returns A and hA unchanged
    5. if A and hA have inconsistent shapes, raises ValueError
    6. if rtol or atol are negative, raises ValueError
    7. if A is not 3D array, raises ValueError
    8. if hA is not 1D array, raises ValueError
    9. if A.shape[2] != hA.shape[0], raises ValueError

    Examples
    --------
    >>> import numpy as np
    >>> from tdcpy.common.compress import compress_matrices_delays
    >>> A0 = np.array([[0, 1],[1, 0]])
    >>> A1 = np.array([[1, 0],[0, 0]])
    >>> A2 = np.array([[0, 0],[0, 1]])
    >>> A = np.stack((A0, A1, A2), axis=2)
    >>> hA = np.array([0., 2., 0.])
    >>> compress_matrices_delays(A, hA)
    (array([[[0., 1.],
        [1., 0.]],
       [[1., 0.],
        [1., 0.]]]), array([0., 2.]))
    """
    # Consider adding tests here - TODO

    unique_hA = np.unique(hA) # sorted in ascending order
    compressed_A = np.zeros(shape=(A.shape[0], A.shape[1], unique_hA.shape[0]))
    
    # iterate over all unique delays and populate matrix A
    for i in range(unique_hA.shape[0]):
        mask = hA == unique_hA[i] # create mask
        compressed_A[:,:,i] = np.sum(A[:,:,mask], axis=2)
    
    # perform elimination of compressed_A[:,:,i] close to zero matrix
    mask = np.all(np.isclose(compressed_A, 0.0, rtol=rtol, atol=atol), axis=(0,1))
    compressed_A = compressed_A[:,:,~mask]
    compressed_hA = unique_hA[~mask]

    return compressed_A, compressed_hA

def compress_bool_matrices_delays(A: npt.NDArray, hA: npt.NDArray, keep_zeros=False):
    """ Compresses boolean matrices - delays representation
    
    Removes delay duplicates, and sorts delays into ascending order,
    i.e. converts the representation (A, hA):

    .. math::
        A_0 x(t - h_{A,0}) + ... + A_{mA} x(t - h_{A,mA})
    
    into the representation (A*, hA*), where:
        
    1. matrices A*[i] are NOT zero
    2. hA* does not contain duplicates

    Parameters
    ----------
    A : array
        3D array of stacked boolean matrices (axis 2)
    hA : array
        1D array (vector) of delays
    keep_zeros : bool
        whether to keep zero matrices in the compressed representation,
        default False
    
    Returns
    -------
    tuple
        A tuple containing

        compressed_A : array
            compressed representation of A
        compressed_hA : array
            compressed vector of delays
    
    Notes
    -----
    1. the sort is "stable" (see numpy.argsort implementation)
    2. if all matrices are zero, returns empty array with shape `(A.shape[0], A.shape[1], 0)` and empty hA array
    3. if A is empty, returns A and hA unchanged
    4. if hA is empty, returns A and hA unchanged
    5. if A and hA have inconsistent shapes, raises ValueError
    6. if A is not 3D array, raises ValueError
    7. if hA is not 1D array, raises ValueError
    8. if A.shape[2] != hA.shape[0], raises ValueError  
    
    Examples
    --------
    >>> import numpy as np
    >>> from tdcpy.common.compress import compress_bool_matrices_delays
    >>> A0 = np.array([[0, 1],[1, 0]])
    >>> A1 = np.array([[1, 0],[0, 0]])
    >>> A2 = np.array([[0, 0],[0, 1]])
    >>> A = np.stack((A0, A1, A2), axis=2)
    >>> hA = np.array([0., 2., 0.])
    >>> compress_bool_matrices_delays(A, hA)
    (array([[[0., 0.],
        [1., 0.]],
       [[1., 0.],
        [1., 0.]]]), array([0., 2.]))
    """
    # Consider adding tests here - TODO

    unique_hA = np.unique(hA) # sorted in ascending order
    compressed_A = np.zeros(shape=(A.shape[0], A.shape[1], unique_hA.shape[0]))
    
    # iterate over all unique delays and populate matrix A
    for i in range(unique_hA.shape[0]):
        mask = hA == unique_hA[i] # create mask
        compressed_A[:,:,i] = np.any(A[:,:,mask], axis=2)
    
    if not keep_zeros:
        # perform elimination of compressed_A[:,:,i] equal to zero matrix
        mask = np.all(compressed_A == False, axis=(0,1))
        compressed_A = compressed_A[:,:,~mask]
        compressed_hA = unique_hA[~mask]

    return compressed_A, compressed_hA

def sort_matrices_delays(A: npt.NDArray, hA: npt.NDArray):
    """ Sorts matrices - delays representation (ascending order by delays)
    
    Sorts delays into ascending order, i.e. changes the representation (A, hA):

    .. math::
        A_0 x(t - h_{A,0}) + ... + A_{mA} x(t - h_{A,mA})                     
    
    into representation (A*, hA*), where:

    1. hA* is now in ascending order
    2. shapes are preserved

    Parameters
    ----------
    A : array
        3D array of stacked matrices (axis 2)
    hA : array
        1D array (vector) of delays

    Returns
    -------
    tuple
        A tuple containing

        compressed_A : array
                compressed representation of A
        compressed_hA : array
                compressed vector of delays

    Notes
    -----
    1. the sort is "stable" (see numpy.argsort implementation)
    2. if A is empty, returns A and hA unchanged
    3. if hA is empty, returns A and hA unchanged
    4. if A and hA have inconsistent shapes, raises ValueError
    5. if A is not 3D array, raises ValueError
    6. if hA is not 1D array, raises ValueError
    7. if A.shape[2] != hA.shape[0], raises ValueError
    8. if hA is already sorted, returns A and hA unchanged

    Examples
    --------
    >>> import numpy as np
    >>> from tdcpy.common.compress import sort_matrices_delays
    >>> A0 = np.array([[0, 1],[1, 0]])
    >>> A1 = np.array([[1, 0],[0, 0]])
    >>> A2 = np.array([[0, 0],[0, 1]])
    >>> A = np.stack((A0, A1, A2), axis=2)
    >>> hA = np.array([0., 2., 0.])
    >>> sort_matrices_delays(A, hA)
    (array([[[0, 0, 1],
            [1, 0, 0]],
    <BLANKLINE>
            [[1, 0, 0],
            [0, 1, 0]]]), array([0., 0., 2.]))
    """
    # Consider adding tests here - TODO
    sorted_index = np.argsort(hA, kind="stable")
    return A[:,:,sorted_index], hA[sorted_index]

def compress_ddae(E: npt.NDArray, A: npt.NDArray, hA: npt.NDArray,
                  B: npt.NDArray, hB: npt.NDArray,
                  C: npt.NDArray, hC: npt.NDArray,
                  D: npt.NDArray, hD: npt.NDArray,
                  rtol=1e-5, atol=1e-8):
    """ Compresses DDAE representation
    """
    raise NotImplementedError(".")


if __name__ == "__main__":
    import doctest
    doctest.testmod()