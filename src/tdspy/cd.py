"""
Strong spectral abscissa of the associated delay difference equation
--------------------------------------------------------------------
TODO
"""

import logging

import numpy as np
import numpy.typing as npt

from .ddae import DDAE

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
        TODO
    
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
        diff = tds
    else:
        diff = tds.get_delay_difference_equation()
    
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





