"""
Set of function for representation compressions
-----------------------------------------------

compression := obtaining minimal sorted representation of "something"

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

        A[0] x(t-hA[0]) + ... + A[mA] x(t-hA[mA])                           (1)
    
    into representation (A*, hA*), where:
        1. matrices A*[i] are NOT close to zero
        1. hA* does not contain duplicates

    Args:
        A: (array): 3D array of stacked matrics (axis 2)
        hA: (array): 1D array (vector) of delays
        rtol (float): relative tolerance for determining matrix element is
            zero, default 1e-5
        atol (float): absolute tolerance for determining matrix element is 
            zero, default 1e-8
    
    Returns:
        tuple containing:

            - compressed_A (array): compressed representation of A
            - compressed_hA (array): compressed vector of delays
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

def sort_matrices_delays(A: npt.NDArray, hA: npt.NDArray):
    """ Sorts matrices - delays representation (ascending order by delays)
    
    Sorts delays into ascending order, i.e. changes the representation (A, hA):

        A[0] x(t-hA[0]) + ... + A[mA] x(t-hA[mA])                           (1)
    
    into representation (A*, hA*), where:
        1. hA* is now in ascending order
        2. shapes are preserved

    Args:
        A: (array): 3D array of stacked matrics (axis 2)
        hA: (array): 1D array (vector) of delays
    
    Returns:
        tuple containing:

            - compressed_A (array): compressed representation of A
            - compressed_hA (array): compressed vector of delays
    
    Notes:
        1. the sort is "stable" (see numpy.argsort implementation)
    """
    # Consider adding tests here - TODO
    sorted_index = np.argsort(hA, kind="stable")
    return A[:,:,sorted_index], hA[sorted_index]

def compress_ddae():
    raise NotImplementedError(".")
