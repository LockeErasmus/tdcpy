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
from .gamma_r import gamma_normalized_diff, GammaInfo

logger = logging.getLogger(__name__)

# DEFINITIONS OF METADATA CLASSES
CDInfo = GammaInfo # TODO fill with metadata

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
        gamma_kwargs: kwargs passed to function `gamma_normalized_diff`
    
    Returns:
        tuple containing:

            - fval (float): f()
            - df (array): derivative of f(.)
    """

    gamma_r, gamma_info = gamma_normalized_diff(DD, hDD, r, **gamma_kwargs)
    th, M, s, u, v = gamma_info # "th", "M", "s", "u", "v"

    # Evaluate numerator
    num = np.sum(
       (
           np.einsum( # evalueate SUM uH @ Di @ v 
                'ijk,jn->ink',
                np.einsum('ni,ijk->njk', np.conj(u)[np.newaxis, :], DD),
                v[:, np.newaxis],
            )
            * hDD * np.exp(-r*hDD) * np.exp(1j*th)
        )
    )
    df = -np.real( np.conj(s) * np.ravel(num) / np.inner(np.conj(u), v) ) / gamma_r
    fval = gamma_r - 1
    return fval, df

class CdRootProblem:

    root_options = {
        "jac": True,
        "scipy_root_method": "hybr",
        "scipy_root_tol": None,
        "scipy_root_callback": None,
        "scipy_root_options": None,

    }
    gamma_kwargs = {}

    def __init__(self, DD, hDD):
        self._DD = DD
        self._hDD = hDD

        self._gamma_info_star = None
        self._fval_star = np.inf

    def fun(self, r) -> tuple[float, float]:

        gamma_r, gamma_info = gamma_normalized_diff(self._DD, self._hDD, r, **self.gamma_kwargs)
        th, M, s, u, v = gamma_info # "th", "M", "s", "u", "v"

        # Evaluate numerator
        num = np.sum(
        (
            np.einsum( # evalueate SUM uH @ Di @ v 
                    'ijk,jn->ink',
                    np.einsum('ni,ijk->njk', np.conj(u)[np.newaxis, :], self._DD),
                    v[:, np.newaxis],
                )
                * self._hDD * np.exp(-r*self._hDD) * np.exp(1j*th)
            )
        )
        df = -np.real( np.conj(s) * np.ravel(num) / np.inner(np.conj(u), v) ) / gamma_r
        fval = gamma_r - 1

        if abs(fval) < self._fval_star: # current best r -> save metadata from gamma computation
            self._gamma_info_star = gamma_info

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
                `gamma_normalized_diff` (see documentation)
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

    if DD.shape[1] == 0:
        # case one delay -> cd = -INF
        return -np.inf, None # TODO

    if DD.shape[2] == 1:
        # case only one delay: the strong spectral abscissa is equal to
        #   cd = ln( rho(DD[0]]) ) / hDD[0]
        gamma0, info = gamma_normalized_diff(DD, hDD, 0, **gamma_kwargs)
        cd = np.log(gamma0) / hDD[0]
        return cd, info # TODO info return, as of now None

    # gamma(r) == 0, degenerate case
    gamma0, info = gamma_normalized_diff(DD, hDD, 0, **gamma_kwargs)
    if gamma0 == 0.0:
        return -np.inf, None # TODO metadata
    
    # gamma(r) =/= 0, use fsolve to find zero crossings of f(r) = gamma(r) - 1

    problem = CdRootProblem(DD, hDD)
    sol = optimize.root(
        fun=problem.fun,
        x0=cd0,
        jac=True,
        method=kwargs.get("scipy_root_method", "hybr"),
        tol=kwargs.get("scipy_root_tol", None),
        callback=kwargs.get("scipy_root_callback", None),
        options=kwargs.get("scipy_root_options", None),
    )

    logger.debug(f"Root Problem solved via `scipy.optimize.root` with settings: {problem.root_options}")
    logger.debug(f"Solution\n---------\n{sol}")
    logger.debug(f"Corresponding {problem._gamma_info_star}")

    # logic for handling solution
    if not sol.success: # i.e. root-finding algorithm failed
        logger.warning("Failed to find zero crossings")
        # TODO log some additional info why fail?
        if gamma0 >= 1:
            cd_star = np.inf
        else:
            cd_star = -np.inf
        raise NotImplementedError(f"NO gamma_info")
    else:
        cd_star = float(sol.x)
        cd_info = problem._gamma_info_star
    
    return cd_star, cd_info
