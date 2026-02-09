"""
Implementation of tds_roots
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .rdde import RDDE
from .ddae import DDAE
from .ndde import NDDE
from .utils import validate_rectangular_region
from .common.compress import compress_matrices_delays
from .stability.characteristic_roots import roots_ddae, RootsInfo

logger = logging.getLogger(__name__)

def roots(tds: RDDE | NDDE | DDAE , r=0.0, **kwargs) -> tuple[npt.NDArray, RootsInfo]:
    """ Computes the characteristic roots of a time-delay system in a given
    right half-plane or rectangular region.

    Args:
        tds (TDS): instance of time-delay system, i.e., RDDE, NDDE or DDAE
        r (int or list): region, specify r as number on real axis or rectangular
            region via 4 coordinates [Re_min, Re_max, Im_min, Im_max], default r=0.0
        kwargs:
            discretization (int): discretization for discretizing DDAE into DAE,
                optional, default None, if not specified, heuristic from [1]
                will be used to obtain sufficient discretization
            max_size_evp (int): maximum allowed size of eigenvalue problem (EVP)
                optional, default 600
            base_delay (float): base delay in case delays are commensurate,
                optional, default None, used in discretization heuristic
    
    Returns:
        tuple containing

            - cr (array): vector of obtained roots
            - roots_info (RootsInfo): Roots metadata containing:
                discretization (int): discretization used for obtaining EVP
                gamma_r_exceeds_one (bool): flag indicating gamma(r) > 1
                index_exceeds_one (bool): flag that index exceeds one
                discretization_eigenvalues (array): eigenvalues of EVP
                max_size_evp_enforced (bool): flag if maximum size of EVP was
                    enforced
                newton_inital_guesses (array): roots before newton corrections
                newton_final_values (array): roots after newton corrections
                newton_residuals (array): newton residuals
                newton_unconverged_initial_guesses (array): mask of unconverged
                    newton initial guesses
                newton_large_corrections (array): mask of "large" corrections
    
    References:

    [1] Wu, Z. and Michiels, W. (2012). Reliably computing all
        characteristic roots of delay differential equations in a given
        right half plane using a  spectral method. Journal of
        Computational and Applied Mathematics, 236(9), pp. 2499-2514.
    [2] Michiels, W. (2011). Spectrum-based stability analysis and 
        stabilisation of systems described by delay differential 
        algebraic equations. IET control theory & applications, 5(16), 
        pp. 1829-1842. 

    """
    if isinstance(r, (int, float)):
        pass # valid definition of RHP
    else: # assume rectangular region -> validate
        r = validate_rectangular_region(r) # raises error if not valid rectangular region
    
    # type of TDS, RDDE and DDAE -> OK, NDDE -> convert to DDAE
    if isinstance(tds, NDDE):
        tds = tds.to_ddae()

    # unpack TDS object
    n = tds.n
    E = tds.E

    if tds.mA == 0:
        hA = np.array([0.0], dtype=E.dtype)
        A = np.zeros(shape=(n,n,1), dtype=E.dtype)
    elif tds.hA[0] > 0:
        hA = np.r_[0, hA] # prepend 0.0 delay
        A = np.concatenate([np.zeros(shape=(n,n,1), dtype=E.dtype), A], axis=2)
    else:
        hA = tds.hA
        A = tds.A
        
    # find all roots via discretization
    cr, cr_info = roots_ddae(E, A, hA, r, **kwargs)

    return cr, cr_info
