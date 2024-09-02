"""
Stabopt utils
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg

logger = logging.getLogger(__name__)

def check_diff(E: npt.NDArray, B: npt.NDArray, C: npt.NDArray, **kwargs) -> bool:
    """ Checks if delay difference equation is dependent on controller
    parameters

    Args:
        TODO
        **kwargs
    
    Returns:
        TODO
    """

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
    
