"""
Strong spectral abscissa of the associated delay difference equation
--------------------------------------------------------------------
TODO
"""

import logging

import numpy as np
import numpy.typing as npt

from .ddae import DDAE
from .common.delay_difference_equation import normalize_diff

from .stability.gamma_r import compute_gamma

logger = logging.getLogger(__name__)



def cd(tds: DDAE, **kwargs):
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
            tol (float): tolerance, optional, default 1e-10
            n_theta (int): the number of grid points in each direction used to
                predict gamma(r), optional, default 10
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
    tol = kwargs.get("tol", 1e-10)
    n_theta = kwargs.get("n_theta", 10)
    cd0 = kwargs.get("cd0", 0)

    assert isinstance(tol, (float, int)) and tol > 0
    assert isinstance(n_theta, int) and n_theta > 0
    assert isinstance(cd0, (float, int))

    # step 1: obtain delay difference equation
    # TODO check tds is correct instance of class
    if tds.is_delay_difference_equation:
        diff = tds.compress()
    else:
        diff = tds.get_delay_difference_equation()
        diff.compress(inplace=True)
    
    # step 2: filter out trivial cases
    if diff is None: # no associated delay difference equation (or empty)
        return -np.inf
    
    if diff.mA < 2:
        # associated delay difference equation empty or one (invertible)
        # term corresponding to a zero delay
        return -np.inf
    
    # step 3: computation
    if diff.mA >= 4:
        logger.warning(("Large number of delays in difference equation. "
                        "Computation of CD might be slow."))
    
    # step 3.1 normalize delay difference equation - continue line 135
    DD, hDD = normalize_diff(diff.A, diff.hA)

    if DD.shape[2] == 1:
        # case only one delay: the strong spectral abscissa is equal to
        #   cd = ln( rho(DD[0]]) ) / hDD[0]
        gamma0, _ = compute_gamma(DD, hDD, 0)
        cd = np.log(gamma0) / hDD[0]
        # TODO info return, as of now None
        return cd, None
    
    # to exclude degenerate case gamma(r) == 0 for all r
    gamma0, _ = compute_gamma(DD, hDD, 0)
    if gamma0 == 0.0:
        return -np.inf, None
    
    # gamma(r) =/= 0 --> use fsolve to find zero crossings of gamma(r)-1
    # TODO continue line 171

    print("passing")


