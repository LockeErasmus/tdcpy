# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

"""
Stabopt utils
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg

logger = logging.getLogger(__name__)

def diff_dependency_mask(Kmask: npt.NDArray, uE: npt.NDArray, vE: npt.NDArray, B: npt.NDArray, C: npt.NDArray, **kwargs) -> npt.NDArray:
    """ Creates dependency mask for coefficients of delay difference eqations
    on controller parameters

    Parameters
    -----------
    Kmask : npt.NDArray
        3d boolean array of shape (p, q, hK) where p is the number of inputs, q the number of outputs and hK the number of controller delays. Kmask[:,:,i] is the mask for the i-th controller term.
    uE : npt.NDArray
        left null space of E matrix of the delay difference equation
    vE : npt.NDArray
        right null space of E matrix of the delay difference equation
    B : npt.NDArray
        input matrix of the delay difference equation
    C : npt.NDArray
        output matrix of the delay difference equation
    **kwargs: additional arguments, currently not used

    Returns
    --------
    npt.NDArray
        3d boolean array of shape (p, q, hK) where p is the number of inputs, q the number of outputs and hK the number of controller delays. 
        Kmask[:,:,i] is the mask for the i-th controller term, indicating which
        coefficients of the delay difference equation depend on the controller
        parameters.

    Notes
    -----
    The dependency mask is computed by evaluating the expression
        uE.T @ B @ K[:,:,i] @ C @ vE
    for each controller term i, where K[:,:,i] is the mask for the i-th controller term.
    If the result is non-zero, the corresponding coefficients of the delay difference
    equation depend on the controller parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from tdcpy.stabopt.utils import diff_dependency_mask
    >>> Kmask = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=bool)
    >>> uE = np.array([[1], [0]])
    >>> vE = np.array([[1], [0]])
    >>> B = np.array([[1, 0], [0, 1]])
    >>> C = np.array([[1, 0], [0, 1]])
    >>> diff_dependency_mask(Kmask, uE, vE, B, C)
    array([[[ True, False],
            [False,  True]]])
    """
    # TODO what if uE, vE empty
    rtol = kwargs.get("rtol", 1e-10)
    atol = kwargs.get("rtol", 1e-10)

    K_mask_3d = np.einsum(# more efficient way to obtain B @ K[:,:,i] @ C for all i
        'ijk,jn->ink',
        np.einsum('ni,ijk->njk', (uE.T).astype(bool) @ B.astype(bool), Kmask),
        C.astype(bool) @ vE.astype(bool),
    )
    return K_mask_3d


def check_diff(E: npt.NDArray, B: npt.NDArray, C: npt.NDArray, **kwargs) -> bool:
    """ Checks if delay difference equation is dependent on controller
    parameters

    Args:
        TODO
        **kwargs
    
    Returns:
        TODO
    """
    raise NotImplementedError("Do not use this function, it is not correct and will be deleted.")

    uE = kwargs.get('uE', None)
    vE = kwargs.get('vE', None)
    rcond = kwargs.get("rcond", 1e-12)
    rtol = kwargs.get("rtol", 1e-10)
    atol = kwargs.get("rtol", 1e-10)

    if uE is None:
        uE = linalg.null_space(E.T, rcond=rcond)
    if vE is None:
        vE = linalg.null_space(E, rcond=rcond) # TODO, SVD is now calculated twice

    # Case where E is non-singular -> DIFF not affected by controller parameter
    if uE.size == 0 or vE.size == 0:
        return False
    
    # check if matrix uE.T @ B @ K[:,:,i] @ C @ vE is close to 0
    if (np.allclose(uE.T @ B, 0, rtol=rtol, atol=atol)
        or np.allclose(C @ vE, 0, rtol=rtol, atol=atol)):
        return False

    return True 
    
