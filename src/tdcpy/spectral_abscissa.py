# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

"""
Spectral abscissa, strong spectral abscissa
-------------------------------------------
Functions:
    1.  spectral_abscissa: computes spectral abscissa of TDS
    2.  spectral_abscissa_diff: computes spectral abscissa of associated delay
        difference equation of TDS
    3.  strong_spectral_abscissa: computes strong spectral abscissa of TDS
    4.  sa: alias for `spectral_abscissa`
    5.  cd: alias for `spectral_abscissa_diff`
    6.  strong_sa: alias for `strong_spectral_abscissa`
"""

from collections import namedtuple
import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .rdde import RDDE
from .ddae import DDAE
from .ndde import NDDE
from .stability.characteristic_roots import rightmost_root
from .stability.gamma_r import gamma_normalized_diff
from .stability.spectral_abscissa import spectral_abscissa_diff as sa_diff

from .common.delay_difference_equation import normalize_diff

logger = logging.getLogger(__name__)

def spectral_abscissa(tds: RDDE | NDDE | DDAE , r=0.0, **kwargs) -> float:
    """ Computes the spectral abscissa of a time-delay system

    Spectral abscissa is defined as

        abscissa = supremum {Re(z): |M(z)| = 0},                            (1)
    
    i.e.,  the smallest upper bound for the real part of the roots of
    characteristic polynomial |M(z)|.

    Args:
        tds (TDS): instance of time-delay system, i.e., RDDE, NDDE or DDAE
        r (float): r is number defining right-half complex plane for
            characteristic roots computation
        **kwargs:
            check_gamma_sa (bool): if true, gamma(sa) < 1+eps check is performed
                in case tds is NDDE or DDAE, default True
            rightmost_root_kwargs (dict): kwargs for `rightmost_root` function,
                default empty dict

    Returns:
        abscissa (float): spectral abscissa
    """

    # tds instance check - TODO, these asserts are not correct
    # assert isinstance(tds, (RDDE, NDDE, DDAE)), "Provided tds has to be RDDE, NDDE or DDAE" 
    # assert tds.is_lti, "provided tds has to be linear time-invariant (LTI)"
    # assert tds.is_real, "provided tds has to contain only real-valued matrices"
    # assert not tds.is_logical, "provied tds can not be logical"
    # assert tds.n > 0, "the dimension of the state variable of tds must be larger than 0"

    # r check
    assert isinstance(r, (float, int)), "r has to be float or int"

    # unpack kwargs
    check_gamma_sa = kwargs.get("check_gamma_sa", True)
    rightmost_root_kwargs = kwargs.get("rightmost_root_kwargs", dict())

    # type of TDS, RDDE and DDAE -> OK, NDDE -> convert to DDAE
    if isinstance(tds, NDDE):
        tds = tds.to_ddae()
    
    # compress tds (this also sorts via delays)
    tds = tds.compress() 

    # obtain right-most root, real part is abscissa
    rmr, rmr_info = rightmost_root(tds.E, tds.A, tds.hA, r=r, **rightmost_root_kwargs)
    abscissa = np.real(rmr) # spectral abscissa as real part of right most root

    if not rmr_info.found or rmr_info.max_size_evp_enforced:
        logger.warning("Spectral abscissa might be inaccurate. Provide a better value for r")
    
    if check_gamma_sa and isinstance(tds, (NDDE, DDAE)):
        # gamma_sa, gamma_info = gamma_normalized_diff() # TODO
        gamma_sa = 1.0 # TODO TODO
        if gamma_sa > 1 + 1e-6:
            logger.warning("Strong spectral abscissa might be larger than spectral abscissa")

    return abscissa

def spectral_abscissa_diff(tds: RDDE | NDDE | DDAE, **kwargs):
    """ Computes strong spectral abscissa of associated delay difference
    equation
 
    This function is based on the expressions in [1, Definition 1.32]
    and [1, Proposition 1.51] for the strong spectral abscissa of delay
    difference equations associated with NDDEs and DDAEs, respectively. 
 
    Args:
        tds (TDS): instance of time-delay system, i.e., RDDE, NDDE or DDAE
        **kwargs:
            cd0 (float): initial guess for the strong spectral abscissa of the
                underlying delay difference equation, optional, default 0.0

    Returns:
        tuple containing:
            -   cd (float): strong spectral abscissa of associated delay
                difference equation
            -   info (TODO): TODO - named tuple matching matlab behaviour?
    
    Notes:
        1. If the associated delay difference equations has a large
           number of delays, the computation of the corresponding strong
           spectral abscissa may be (very) slow.

    References
    ----------
    [1] Michiels, W. and Niculescu, S.I. (2014). Stability, control, and
        computation for time-delay systems: an eigenvalue-based approach. 
        Society for Industrial and Applied Mathematics (SIAM), Philadelphia, PA
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
    cd_star, cd_info = sa_diff(DD, hDD, **kwargs) # TODO I do not like this name convention

    return cd_star, cd_info

def strong_spectral_abscissa(tds: RDDE | NDDE | DDAE , r=0.0, **kwargs) -> float:
    """ Computes the strong spectral abscissa of a time-delay system

    Strong spectral abscissa is defined as

        strong_abscissa = MAX (CD, abscissa),
    
    where

        CD ... strong spectral abscissa of associated delay-difference eq.
        abscissa = supremum {Re(z): |M(z)| = 0},

        sa = supremum {Re(z): |M(z)| = 0},                                  (1)
    
    i.e.,  the smallest upper bound for the real part of the roots of
    characteristic polynomial |M(z)|.

    Args:
        tds (TDS): instance of time-delay system, i.e., RDDE, NDDE or DDAE
        r (float): r is number defining right-half complex plane for characteristic roots computation
        **kwargs:
            check_gamma_sa (bool): if true, gamma(sa) < 1+eps check is performed
                in case tds is NDDE or DDAE, default True
            rightmost_root_kwargs (dict): kwargs for `rightmost_root` function,
                default empty dict

    Returns:
        strong_sa (float): spectral abscissa
    """
    sa = spectral_abscissa(tds, r=r, **kwargs)
    cd, cd_info = spectral_abscissa_diff(tds, r=r, **kwargs) # TODO fix info
    return max(sa, cd)


def sa(*args, **kwargs):
    """ Alias for `spectral_abscissa` """
    return spectral_abscissa(*args, **kwargs)

def cd(*args, **kwargs):
    """ Alias for `spectral_abscissa_diff` """
    return spectral_abscissa_diff(*args, **kwargs)

def strong_sa(*args, **kwargs):
    """ Alias for `strong_spectral_abscissa` """
    return strong_spectral_abscissa(*args, **kwargs)

    


