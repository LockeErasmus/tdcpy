"""
Delay controller design (stabilization via BFGS)
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg, optimize

from tdspy.stability.characteristic_roots import rightmost_root, RightmostRootInfo
from tdspy.common.compress import compress_matrices_delays
from .gradients import func_sa, func_cd, func_gamma

logger = logging.getLogger("__name__")

def design_bfgs(E: npt.NDArray, P:npt.NDArray, hP:npt.NDArray, K0, hK, B, C, **kwargs):
    """
    Args:
        TODO
        **kwargs:
            mask (array): masking gradient
    """

    Kmask = kwargs.get("mask", np.full_like(K0, fill_value=True, dtype=bool))
    sol = optimize.minimize(
        func_sa,
        K0.reshape(-1),
        args=(E, P, hP, hK, Kmask, B, C),
        jac=True,
        method="L-BFGS-B",
        options=kwargs.get("options", {}),
    )
    return sol