"""
Spectral abscissa
-----------------
We distinguish between three types:

1. spectral abscissa (denote `alpha`)
2. spectral abscissa of delay-difference equation (denote `cd`)
3. strong spectral abscissa := MAX(alpha, cd)
"""

from collections import namedtuple
import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg, optimize

from tdspy.common.delay_difference_equation import normalize_diff
from .gamma_r import compute_gamma

logger = logging.getLogger(__name__)

# DEFINITIONS OF METADATA CLASSES
CDInfo = namedtuple("CDInfo", ["a"]) # TODO fill with metadata

def spectral_abscissa():
    raise NotImplementedError(".")

def func(r, DD, hDD, gamma_kwargs):
    """ Value and derivative of a function

        f(r) = gamma(r) - 1
    
    f(r) = 0 corresponds to zero crossings
    
    Note that gamma(r) is gamma evaluated at r of delayed difference equation
    defined via DD, hDD.
    
    Args: TODO
        r 
        DD
        hDD
        gamma_kwargs: kwargs passed to function `compute_gamma`
    
    Returns:
        tuple containing:

            - fval (float): f()
            - df (array): derivative of f(.)
    """

    gamma_r, gamma_info = compute_gamma(DD, hDD, r, gamma_kwargs)
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

def spectral_abscissa_diff(DD: npt.NDArray, hDD: npt.NDArray, **kwargs) -> tuple[float, CDInfo]:
    """ Computes strong spectral abscissa of normalized delay difference
    equation
 
    This function is based on the expressions in [1, Definition 1.32]
    and [1, Proposition 1.51] for the strong spectral abscissa of delay
    difference equations associated with NDDEs and DDAEs, respectively. 
 
    References:
    [1] Michiels, W. and Niculescu, S.I. (2014). Stability, control, and
        computation for time-delay systems: an eigenvalue-based approach. 
        Society for Industrial and Applied Mathematics (SIAM), Philadelphia, PA

    Args:
        DD (array): TODO, NORMALIZED
        hDD (array): TODO, NORMALIZED
        **kwargs:
            cd0 (float): initial guess for the strong spectral abscissa of the
                underlying delay difference equation, optional, default 0.0
            scipy_root_method (str): scipy.optimize.root method, default 'hybr',
            scipy_root_tol (float): Tolerance for termination. For detailed
                control, use `scipy_root_options`, default None
            scipy_root_callback (function): Optional callback function. It is
                called on every iteration as `callback(x, f)` where x is the
                current solution and f the corresponding residual. For all
                methods but 'hybr' and 'lm'.
            scipy_root_options (dict): a dictionary of solver options (method),
                default None
            gamma_kwargs (dict): keyword arguments passed to function
                `compute_gamma` (see documentation)
    Returns:
        tuple containing:

            - cd (float): strong spectral abscissa of associated delay
                difference equation
            - info (TODO): TODO - named tuple matching matlab behaviour?
    """
    cd0 = kwargs.get("cd0", 0)
    gamma_kwargs = kwargs.get("gamma_kwargs", dict())

    assert hDD.size > 0 and DD.size > 0, "empty delay-difference equation not allowed"
    assert hDD.shape[0] > 0, "empty delay difference equation not allowed"
    assert hDD.shape[0] == DD.shape[2], "len of DD and hDD has to match"

    if DD.shape[2] == 1:
        # case only one delay: the strong spectral abscissa is equal to
        #   cd = ln( rho(DD[0]]) ) / hDD[0]
        gamma0, _ = compute_gamma(DD, hDD, 0, **gamma_kwargs)
        cd = np.log(gamma0) / hDD[0]
        return cd, None # TODO info return, as of now None

    # gamma(r) == 0, degenerate case
    gamma0, _ = compute_gamma(DD, hDD, 0, **gamma_kwargs)
    if gamma0 == 0.0:
        return -np.inf, None # TODO metadata
    
    # gamma(r) =/= 0, use fsolve to find zero crossings of f(r) = gamma(r) - 1
    sol = optimize.root(
        fun=func,
        x0=cd0,
        args=(DD, hDD, gamma_kwargs),
        jac=True,
        method=kwargs.get("scipy_root_method", "hybr"),
        tol=kwargs.get("scipy_root_tol", None),
        callback=kwargs.get("scipy_root_callback", None),
        options=kwargs.get("scipy_root_options", None),
    )

    # logic for handling solution
    if not sol.success: # i.e. root-finding algorithm failed
        logger.warning("Failed to find zero crossings")
        # TODO log some additional info why fail?
        if gamma0 >= 1:
            cd_star = np.inf
        else:
            cd_star = -np.inf
    else:
        cd_star = float(sol.x)
    
    return cd_star, None # TODO metadata
