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

def func(r, DD, hDD, gamma_kwargs):
    """ Evaluates CD(.) and its derivative


    Function:

        f = gamma(r) - 1,                                           (1)
    
    and its derivative

             - Re { conj(s) num / (u^H * v) }
        df = ------------------------------  ,                      (2)
                        gamma(r)
    
    where

        num = SUM u^H * D[i] * v * hDD[i] * exp(-r*hDD[i]) * exp(1j*th[i]),

    and s, u, v, th are info outputs of compute_gamma.

    Args:
        r (float): TODO
        DD (array): coefficient matrices packed into 3D array shaped (n,n,m),
            note that coefficients for x(t) are assumed to be identity matrix
            and therefore omitted (see functions for converting DDAE to delay
            difference equation and normalizing)
        hDD (array): delays represented by 1D array shaped (m,), note that delay
            0 is omitted
        gamma_kwargs (dict): **kwargs fed into `compute_gamma` function
        
    Returns:
        tuple containing

        - f (float): 1D array representing function F evaluated at x
        - df (array): 1D array of jacobian of F evaluated at x
    """

    gamma_r, info = compute_gamma(DD, hDD, r, **gamma_kwargs)
    th, M, s, u, v = info # unpack named tuple
    
    # num = (np.ravel(np.conj(info.u[np.newaxis, ]) @ np.transpose(info.v[np.newaxis, :] @ DD, (1,0,2)))
    #       *hDD * np.exp(-r*hDD)* np.exp(1j*info.th))
    # THIS IS UNREADABLE ^^^
    num = 0
    for i in range(DD.shape[2]):
        num += (np.conj(u[np.newaxis, ] @ DD[:,:,i]) @ v[:, np.newaxis] 
                * hDD[i] * np.exp(-r*hDD[i])* np.exp(1j*th[i]))
    num = np.ravel(num) # get rid of shape (1,1)
    f = gamma_r - 1
    df = - np.real(np.conj(s) * num / np.inner(np.conj(u), v)) / gamma_r
    return f, df

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
        diff.compress(inplace=True) # TODO check for None?
    
    # step 2: filter out trivial cases
    if diff is None: # no associated delay difference equation (or empty)
        return -np.inf
    
    if diff.mA < 2:
        # associated delay difference equation empty or one (invertible)
        # term corresponding to a zero delay
        return -np.inf
    
    # warning when number of delays > 3
    if diff.mA >= 4:
        logger.warning(("Large number of delays in delay difference equation. "
                        "Computation of CD might be slow."))
    
    # step 3: computation
    # step 3.1 normalize delay difference equation - continue line 135
    DD, hDD = normalize_diff(diff.A, diff.hA)

    if DD.shape[2] == 1:
        # CASE 1: only one delay, the strong spectral abscissa is equal to:
        #   cd = ln( rho(DD[0]]) ) / hDD[0]
        logger.debug(("Only one delay detected in strong spectral abscissa of "
                      "delay difference equation and hence will be calculated "
                      "as cd=ln( gamma(0) / tau )"))
        gamma0, gamma_info = compute_gamma(DD, hDD, 0)
        cd = np.log(gamma0) / hDD[0]
        return cd, None # TODO info return
    
    # CASE 2: more than one delay
    logger.debug("")
    # to exclude degenerate case gamma(r) == 0 for all r
    gamma0, gamma_info = compute_gamma(DD, hDD, 0)
    if gamma0 == 0.0:
        return -np.inf, None # TODO info return
    
    # gamma(r) =/= 0 --> use fsolve to find zero crossings of gamma(r)-1
    soln = None # TODO continue line 171

    print("passing")


