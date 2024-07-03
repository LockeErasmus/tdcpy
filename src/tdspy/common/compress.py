"""


"""
import numpy as np
import numpy.typing as npt


def compress_matrices_delays(A: npt.NDArray, hA: npt.NDArray, rtol=1e-5, atol=1e-8):
    """ Removes delay duplicates, sorts delays into ascending order

    Converts:

        A[0] x(t-hA[0]) + ... + A[mA] x(t-hA[mA])
    
    into:

        A

    Such that no duplicates in 

    Args:
        A: array TODO
        hA: array TODO 
        rtol (float): relative tolerance for determining matrix element is
            zero, default 1e-5
        atol (float): absolute tolerance for determining matrix element is 
            zero, default 1e-8
    
    Returns:
        - TODO

        - compressed_A
        - compressed_hA
    """
    # TODO perform tests

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

def compress_ddae():
    raise NotImplementedError
