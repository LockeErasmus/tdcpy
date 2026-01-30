"""
Basic operations for time-delay LTI systems in matrix representation
"""

import numpy as np
import numpy.typing as npt

from .compress import compress_matrices_delays

def shift_ddae(E: npt.NDArray, A: npt.NDArray,
               hA: npt.NDArray, shift: float | complex) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    r""" Shifts the delays of a DDAE reprezentation in Laplace by `shift`.

    Assume DDAE in laplace form

    .. math::
        E s X(s) = \sum_{i=0}^{mA} A_i e^{-h_{A,i} s} X(s)

    where we assume at least 2 (mA>1) sorted unique delays h_{A,i} \geq 0 and h_{A,0}=0.
    We want to perform shift in Laplace domain

    .. math::
        s \leftarrow s + \text{shift}

    
    Parameters
    ----------
    E : array
        LHS matrix of DDAE
    A : array
        3D array of RHS matrices of DDAE
    hA : array
        vector of delays associated with array A
    shift : float | complex
        complex plane shift

    Returns
    -------
    tuple
        A tuple containing

        E_shifted : array
            LHS matrix of shifted DDAE (same as input E)
        A_shifted : array
            3D array of RHS matrices of shifted DDAE (same as input A)
        hA_shifted : array
            vector of delays associated with array A after shifting
    """
    # TODO assume hA unique, non-negative, sorted with hA[0] == 0 len(hA) > 1
    E_shifted = np.copy(E) # TODO, change dtype or not?
    A_shifted = np.copy(A, dtype=A.dtype if shift is float else np.complex128)
    hA_shifted = np.copy(hA)

    A_shifted[:,:,0] -= shift * E_shifted
    A_shifted[:,:,:] = A_shifted * np.exp(-shift * hA_shifted)

    return E_shifted, A_shifted, hA_shifted

def normalize_ddae(E: npt.NDArray, A: npt.NDArray,
                   hA: npt.NDArray, factor: float) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """ Normalizes a DDAE by a given factor.

    TODO math docstring

    Parameters
    ----------
    E : array
        LHS matrix of DDAE
    A : array
        3D array of RHS matrices of DDAE
    hA : array
        vector of delays associated with array A

    Returns
    -------
    tuple
        A tuple containing

        E_normalized : array
            LHS matrix of normalized DDAE (same as input E)
        A_normalized : array
            3D array of RHS matrices of normalized DDAE
        hA_normalized : array
            vector of delays associated with array A after normalization
    """
    if factor == 0:
        raise ValueError("Normalization factor cannot be zero.")

    return E, A * factor, hA / factor


def shift_normalize_ddae(E: npt.NDArray, A: npt.NDArray,
                          hA: npt.NDArray, shift: float | complex, **kwargs) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """ Shifts and normalizes a DDAE such that resulting matrix representation ... TODO

    """
    # TODO assume hA unique, non-negative, sorted with hA[0] == 0 len(hA) > 1

    if len(hA) < 1 or hA[0] != 0:
        raise ValueError("shift_normalize_ddae requires hA with hA[0] == 0")
    
    hA_max = np.max(hA)
    if hA_max == 0:
        raise ValueError("shift_normalize_ddae requires hA with at least one positive delay")

    new_E = np.copy(E)
    new_A = np.copy(A, dtype=A.dtype if shift is float else np.complex128)
    new_hA = hA / hA_max

    return np.copy


    




