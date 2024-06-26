"""
Strong spectral abscissa of the associated delay difference equation
--------------------------------------------------------------------
TODO:
    1. as of now, just make it work, but later separate high level API and
       pure math functions
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg, optimize

from .rdde import RDDE
from .ddae import DDAE
from .ndde import NDDE
from .common.delay_difference_equation import normalize_diff

from .stability.spectral_abscissa import spectral_abscissa_diff, CDInfo

logger = logging.getLogger(__name__)

def cd(tds: RDDE | NDDE | DDAE, **kwargs):
    """ Computes strong spectral abscissa of associated delay difference
    equation
 
    This function is based on the expressions in [1, Definition 1.32]
    and [1, Proposition 1.51] for the strong spectral abscissa of delay
    difference equations associated with NDDEs and DDAEs, respectively. 
 
    References:
    [1] Michiels, W. and Niculescu, S.I. (2014). Stability, control, and
        computation for time-delay systems: an eigenvalue-based approach. 
        Society for Industrial and Applied Mathematics (SIAM), Philadelphia, PA

    Args:
        tds (TODO): TODO
        **kwargs:
            cd0 (float): initial guess for the strong spectral abscissa of the
                underlying delay difference equation, optional, default 0.0

    Returns:
        tuple containing:

            - cd (float): strong spectral abscissa of associated delay
                difference equation
            - info (TODO): TODO - named tuple matching matlab behaviour?
    
    Notes:
        1. If the associated delay difference equations has a large
           number of delays, the computation of the corresponding strong
           spectral abscissa may be (very) slow.
    """
    cd0 = kwargs.get("cd0", 0)

    assert isinstance(cd0, (float, int))

    # step 1: obtain delay difference equation
    # TODO check tds is correct instance of class ?
    if tds.is_delay_difference_equation:
        diff = tds.compress()
    else:
        diff = tds.get_delay_difference_equation()
        diff.compress(inplace=True)
    
    # step 2: filter out trivial cases
    if diff is None: # no associated delay difference equation (or empty)
        return -np.inf, None # TODO metadata
    
    if diff.mA < 2:
        # associated delay difference equation empty or one (invertible)
        # term corresponding to a zero delay
        return -np.inf, None # TODO metadata
    
    # step 3: computation
    if diff.mA >= 4:
        logger.warning(("Large number of delays in difference equation. "
                        "Computation of CD might be slow."))
    
    # step 3.1 normalize delay difference equation
    DD, hDD = normalize_diff(diff.A, diff.hA)

    # step 3.2 find spectral abscissa of associated normalized DIFF
    cd_star, cd_info = spectral_abscissa_diff(DD, hDD, **kwargs)

    return cd_star, cd_info


