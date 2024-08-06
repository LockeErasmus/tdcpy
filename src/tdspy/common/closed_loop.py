"""
Closed-loop and controller related functionality
------------------------------------------------
"""

import numpy as np
import numpy.typing as npt

import logging

from .composition import concatenate_2x2_by_delays
from .compress import compress_bool_matrices_delays

logger = logging.getLogger(__name__)

def controller_reprezentation(order: int, n_inputs: int, n_outputs: int, hA: npt.NDArray = None, hB: npt.NDArray = None, hC: npt.NDArray = None, hD: npt.NDArray = None, **kwargs):
    """ Creates empty controller reprezentation
    
    TODO
    
    """

    assert isinstance(order, int)
    assert order >= 0
    assert isinstance(n_inputs, int)
    assert n_inputs > 0
    assert isinstance(n_outputs, int)
    assert n_outputs > 0

    n, m = order + n_inputs, order + n_inputs

    dtype = np.float32

    if order == 0:
    # special case for static controller
        if hA is not None or hB is not None or hC is not None:
            # user provided delays for A, B or C matrices -> warning
            logger.warning(("You have specified controller with order 0 but "
            "also provided vector of delays hA, hB or hC. These"
            "inputs will be ignored as the only sensible vector"
            " of delays for static controller is hD."))
        
        hA = np.zeros(shape=(0,), dtype=dtype)
        hB = np.zeros(shape=(0,), dtype=dtype)
        hC = np.zeros(shape=(0,), dtype=dtype)

        A = np.ones(shape=(order, order, 0), dtype=bool)
        B = np.ones(shape=(order, n_inputs, 0), dtype=bool)
        C = np.ones(shape=(n_outputs, order, 0), dtype=bool)

        if hD is None:
            hD = np.array([0.0], dtype=dtype)
            logger.debug(f"No delays specified, setting {hD=}")

        D = np.ones(shape=(n_outputs, n_inputs, hD.shape[0]), dtype=bool)
    else:
        # order > 1 and therefore if user provides delays, all have to have at
        # least one element

        # TODO asserts and defaults
        if hA is None:
            hA = np.array([0.0], dtype=dtype)
        if hB is None:
            hB = np.array([0.0], dtype=dtype)
        if hC is None:
            hC = np.array([0.0], dtype=dtype)
        if hD is None:
            hD = np.array([0.0], dtype=dtype)

        A = np.ones(shape=(order, order, hA.shape[0]), dtype=bool)
        B = np.ones(shape=(order, n_inputs, hB.shape[0]), dtype=bool)
        C = np.ones(shape=(n_outputs, order, hC.shape[0]), dtype=bool)
        D = np.ones(shape=(n_outputs, n_inputs, hD.shape[0]), dtype=bool)
        
    E, K, hK = concatenate_2x2_by_delays(np.eye(order, dtype=dtype), A, B, C, D, hA, hB, hC, hD)
    K, hK = compress_bool_matrices_delays(K, hK)

    return E, K, hK



    
