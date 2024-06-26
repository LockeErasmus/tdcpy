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

from .stability.gamma_r import compute_gamma

logger = logging.getLogger(__name__)


def func(r, DD, hDD, **kwargs):
    """ Value and derivative of a function

        f(r) = gamma(r) - 1
    
    f(r) = 0 corresponds to zero crossings
    
    Note that gamma(r) is gamma evaluated at r of delayed difference equation
    defined via DD, hDD.
    
    Args: TODO
        r 
        DD
        hDD
        **kwargs: kwargs passed to function `compute_gamma`
    
    Returns:
        tuple containing:

            - fval (float): f()
            - df (array): derivative of f(.)
    """

    gamma_r, gamma_info = compute_gamma(DD, hDD, r, **kwargs)
    th, M, s, u, v = gamma_info # "th", "M", "s", "u", "v"

    num = (np.conj(u)[np.newaxis,:] @ DD[:,:,0] @ v[:, np.newaxis]) * hDD[0]*np.exp(-r*hDD[0])*np.exp(1j*th[0])
    for i in range(1, DD.shape[2]):
        num += (np.conj(u)[np.newaxis,:] @ DD[:,:,i] @ v[:, np.newaxis]) * hDD[i]*np.exp(-r*hDD[i])*np.exp(1j*th[i])
    print(f"{num=}")
    print(np.ravel(num))
    df = -np.real( np.conj(s) * np.ravel(num) / np.inner(np.conj(u), v) ) / gamma_r
    print(f"{df=}")
    fval = gamma_r - 1
    return fval, df

f = optimize._optimize.MemoizeJac(func)
fprime = f.derivative


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
    gamma0, _ = compute_gamma(DD, hDD, 0, **kwargs)
    if gamma0 == 0.0:
        return -np.inf, None
    
    # gamma(r) =/= 0 --> use fsolve to find zero crossings of gamma(r)-1
    sol = optimize.root(
        fun=func,
        x0=cd0,
        args=(DD, hDD), # TODO args
        jac=True,
        method="hybr",
    )
   
    if not sol.success: # i.e. root-finding algorithm failed
        logger.warning("Failed to find zero crossings")
        # TODO log some additional info why fail?
        if gamma0 >= 1:
            cd_star = np.inf
        else:
            cd_star = -np.inf
    else:
        cd_star = float(sol.x)

    # TODO check inf

    return cd_star, None # TODO metadata


