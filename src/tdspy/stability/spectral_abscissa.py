"""
Spectral abscissa
-----------------
We distinguish between three types:

1. spectral abscissa (denote ``alpha``)
2. spectral abscissa of delay-difference equation (denote ``cd``)
3. strong spectral abscissa := MAX(alpha, cd)

Functions
---------
- ``spectral_abscissa``: placeholder for classic spectral abscissa routines.
- ``func``: evaluate ``f(r)=gamma(r)-1`` and its derivative at a scalar ``r``.
- ``CdRootProblem``: root-problem wrapper object used by ``scipy.optimize.root``.
- ``spectral_abscissa_diff``: compute the delay-difference contribution ``cd``.

References:
    [1] Michiels, W. and Niculescu, S.I. (2014). Stability, control, and
    computation for time-delay systems: an eigenvalue-based approach.
    Society for Industrial and Applied Mathematics (SIAM), Philadelphia, PA
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
    """Evaluate f(r) = gamma(r) - 1 and its derivative.

    Parameters
    ----------
    r : float
        Real number defining right-half complex plane for
        characteristic roots computation
    DD : npt.NDArray
        Normalized delay-difference matrices defining the operator.
    hDD : npt.NDArray
        Delays corresponding to the slices of ``DD`` (shape must match
        ``DD.shape[2]``).
    gamma_kwargs : dict
        Additional keyword arguments forwarded to
        :func:`gamma_normalized_diff`.

    Returns
    -------
    fval : float
        The scalar value ``gamma(r) - 1``.
    df : float
        The derivative df/dr evaluated at ``r``.

    Notes
    -----
    The implementation calls :func:`gamma_normalized_diff` to obtain
    the normalized spectral radius and associated metadata, then computes
    the derivative using the expression derived from the numerator
    involving the left/right eigenvectors returned by the helper.
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
    """Compute the strong spectral abscissa of a normalized delay-difference equation.

    This routine follows the formulation in Michiels & Niculescu (2014)
    and locates roots of ``f(r)=gamma(r)-1`` using ``scipy.optimize.root``
    to determine the delay-difference contribution to the strong spectral
    abscissa.

    Parameters
    ----------
    DD : npt.NDArray
        Normalized delay-difference matrices of shape ``(n, n, m)``.
    hDD : npt.NDArray
        Delays corresponding to the third dimension of ``DD`` (shape ``(m,)``).
    **kwargs : dict
        Optional keyword arguments:

        - ``cd0`` (float): initial guess for the strong spectral abscissa,
          default ``0.0``.
        - ``scipy_root_method`` (str): method for ``scipy.optimize.root``,
          default ``'hybr'``.
        - ``scipy_root_tol`` (float): tolerance for the root solver.
        - ``scipy_root_callback`` (callable): optional callback ``callback(x, f)``.
        - ``scipy_root_options`` (dict): solver-specific options.
        - ``gamma_kwargs`` (dict): kwargs forwarded to
          :func:`gamma_normalized_diff`.

    Returns
    -------
    cd : float
        The strong spectral abscissa of the associated delay-difference
        equation. May be ``-np.inf`` or ``np.inf`` in degenerate cases.
    info : CDInfo
        Metadata returned from :func:`gamma_normalized_diff` corresponding
        to the root with smallest residual (or ``None`` when unavailable).

        Notes
        -----
        - If ``DD.shape[1] == 0`` the function returns ``-np.inf`` to indicate
            no meaningful contribution from the delay-difference operator.
        - If the root-finding fails but the zero-frequency gamma satisfies
            ``gamma(0) >= 1``, the routine treats the strong spectral abscissa
            as ``np.inf``; otherwise ``-np.inf`` is used. In the current
            implementation a failed root solve raises ``NotImplementedError``
            instead of returning a metadata-rich result.

        References:
        [1] Michiels, W. and Niculescu, S.I. (2014). Stability, control, and
            computation for time-delay systems: an eigenvalue-based approach.
            Society for Industrial and Applied Mathematics (SIAM), Philadelphia, PA
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
