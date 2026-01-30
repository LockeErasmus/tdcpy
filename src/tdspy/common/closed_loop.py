"""
Closed-loop and controller related functionality
------------------------------------------------

Implemented functions:

1. `controller_reprezentation`: creates an empty controller representation ``(E, K, hK)`` given the order, number of inputs and outputs, and delays for matrices ``(Ac, Bc, Cc, Dc)`` (if applicable)

"""

import numpy as np
import numpy.typing as npt

import logging

from .composition import concatenate_2x2_by_delays
from .compress import compress_matrices_delays, compress_bool_matrices_delays

logger = logging.getLogger(__name__)

def controller_reprezentation(order: int, n_inputs: int, n_outputs: int, hA: npt.NDArray = None, hB: npt.NDArray = None, hC: npt.NDArray = None, hD: npt.NDArray = None, **kwargs):
    """ Creates an empty controller reprezentation
    
    Parameters
    -----------
    order : int
        Controller order (0 for static controller)
    n_inputs : int
        Number of controller inputs
    n_outputs : int
        Number of controller outputs
    hA : npt.ndarray, optional
        Vector of delays for A matrix
    hB : npt.ndarray, optional
        Vector of delays for B matrix
    hC : npt.ndarray, optional
        Vector of delays for C matrix
    hD : npt.ndarray, optional
        Vector of delays for D matrix
    **kwargs: dict, optional
        Additional arguments (not used)

    Returns
    -------
    E : npt.NDArray
        Descriptor matrix of the controller
    K : npt.NDArray
        System matrix of the controller
    hK : npt.NDArray
        Vector of delays for K matrix

    Notes
    ------
    Given the order of the controller, number of inputs and outputs, and
    optional delay vectors for matrices A, B, C, D, this function creates
    an empty controller representation of a dynamic controller (order > 0) or
    static controller (order = 0).

    .. math::

        E x' = \sum_{k=0}^{hK} K_k x(t - hK_k)

    where K is constructed for controller matrices A, B, C, D as:
    .. math::

        K = \begin{bmatrix}
            A & B \\
            C & D
        \end{bmatrix}

    and hK is the concatenated vector of delays i.e. :math:`hK = [hA_c \quad hB_c \quad hC_c \quad hD_c]`

    - For static controller (order=0) only hD is used, other delay vectors
      are ignored (and a warning is issued if they are provided)
    - If order > 0 and user does not provide delay vectors, they are set to
      [0.0] by default
    - The resulting controller is fully connected (all entries in A, B, C, D
      matrices are True)
    - The resulting controller is in the form suitable for creating
      ClosedLoop object

    Examples
    --------
    >>> import numpy as np
    >>> from tdspy.common.closed_loop import controller_reprezentation
    >>> E, K, hK = controller_reprezentation(order=1, n_inputs=2, n_outputs=1)
    >>> E.shape
    (2, 3)
    >>> K.shape
    (2, 3, 1)
    >>> hK
    array([0.], dtype=float32)
    >>> E, K, hK = controller_reprezentation(order=0, n_inputs=2, n_outputs=1, hD=np.array([0.0, 1.0]))
    >>> E.shape
    (1, 2)
    >>> K.shape
    (1, 2, 2)
    >>> hK
    array([0., 1.], dtype=float32)

    """

    if not isinstance(order, int):
        raise TypeError("order must be an integer")
    if order < 0:
        raise ValueError("order must be >= 0")
    if not isinstance(n_inputs, int):
        raise TypeError("n_inputs must be an integer")
    if n_inputs <= 0:
        raise ValueError("n_inputs must be > 0")
    if not isinstance(n_outputs, int):
        raise TypeError("n_outputs must be an integer")
    if n_outputs <= 0:
        raise ValueError("n_outputs must be > 0")
    if n_inputs <= 0:
        raise ValueError("n_inputs must be > 0")
    if not isinstance(n_outputs, int):
        raise TypeError("n_outputs must be an integer")
    if n_outputs <= 0:
        raise ValueError("n_outputs must be > 0")

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


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import doctest
    doctest.testmod()


    
