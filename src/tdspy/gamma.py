"""
implementation of tds_gamma_r
"""

import logging

import numpy as np
import numpy.typing as npt

from .rdde import RDDE
from .ddae import DDAE
from .ndde import NDDE

from .common.delay_difference_equation import ddae_to_diff, normalize_diff
from .common.compress import compress_matrices_delays
from .stability.gamma_r import gamma_diff, gamma_normalized_diff, GammaInfo

logger = logging.getLogger()

def gamma(tds: DDAE, r: float,  **kwargs) -> tuple[float, GammaInfo]:
    """ Computes gamma(r) of given time-delay system

    Args:
        tds (TDS): time-delay system
        r (float): point from complex plane
        **kwargs:
            TODO
    
    Returns:
        tuple containning:

            - gamma_r (float): gamma(r) of given TDS
            - metadata (GammaInfo): metadata, containing:
                TODO

    Computation consists of the follwing steps:
        (1) obtain associated delay difference equation (DIFF)
        (2) normalize DIFF, where DIFF takes form:

            0 = x(t) + DD[0] * x(t-hDD[0]) + ... + DD[m-1]*x(t-hDD[m-1]),   (1)

        (3) solve optimzation problem, gamma(r) of (1) is then given by maximum
            of:
                rho( SUM for all k DD[k]*exp(-r*hDD[k])*exp(1j*theta[k]) )
            where rho(.) is spectral radius of its matrix argument, and theta 
            is from [0, 2*pi)^m.

            The optimization problem is solved via Dekker-Brent method (aka
            predictor - corrector), see [1].

    [1] Atkinson, Kendall. An introduction to numerical analysis.
        John wiley & sons, 1991.
    
    Notes:
        1. this problem scales very badly with number of delays as the predictor
            searches through the grid whichs dimensionality corresponds to
            number of delays of normalized DIFF
    """

    if isinstance(tds, NDDE):
        tds = tds.to_ddae() # convert to DDAE form if NDDE
        # TODO better use low level functions here?
    
    # unpack system and compress
    E = tds.E 
    A, hA = compress_matrices_delays(tds.A, tds.hA)
    # obtain DIFF
    D, hD = ddae_to_diff(E, A, hA) # TODO tol, rcond KWARGS

    if hD.shape[0] < 2: # solve trivial case
        g_info = GammaInfo(
            th=np.zeros((0,)),
            M=np.zeros((1,1,hA.shape[0])),
            s=0.+0j,
            u=np.zeros((0,)),
            v=np.zeros((0,)),
        )
        return 0.0, g_info

    DD, hDD = normalize_diff(D, hD)
    g, g_info = gamma_normalized_diff(DD, hDD, r) # TODO KWARGS

    return g, g_info
