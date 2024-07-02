"""
Implementation of tds_roots
"""
from collections import namedtuple
import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .rdde import RDDE
from .ddae import DDAE
from .ndde import NDDE
from .stability.discretization_heuristic import compute_n_rhp, compute_n_rect
from .stability.bounds import lower_bound, upper_bound
from .stability.newton import newton_correction
from .common.discretization import discretize
from .gamma import gamma


from .stability.characteristic_roots import roots_ddae, RootsInfo

logger = logging.getLogger(__name__)

def roots(tds: RDDE | NDDE | DDAE , r=0.0, **kwargs):
    """ Computes the characteristic roots of a time-delay system in a given
    right half-plane or rectangular region.

    Args:
        tds (TDS): instance of time-delay system, i.e., RDDE, NDDE or DDAE
        r (int or list): region, specify r as number on real axis or rectangular
            region via 4 coordinates [Re_min, Re_max, Im_min, Im_max], default r=0.0
        kwargs:
            max_size_evp (int): TODO, default 600
            discretization (int): discretization, if None heuristic is envoked,
                default None, keep default if you don't know, has to be > 1
            basic_delay (float): define if delays are commensurate, default
                None, used in discretization heuristic case `rhp`
    
    Returns:
            tuple containing

                - roots (array): array of found roots
                - metadata (RootsInfo): named tuple consisting of TODO
    """

    # checks for region definition
    if isinstance(r, (int, float)):
        # all OK
        case = "rhp"
    elif isinstance (r, list):
        assert len(r) == 4, "region has to be defined in form [a,b,c,d]"
        assert r[0] < r[1] and r[2] < r[3], "region has to be defined as [a,b,c,d], a<b, c<d"
        assert np.all(~np.isinf(r)), "region has to be finite rectangle"
        case = "rect"
    else:
        raise ValueError(("Region (argument `r`) has to be defined as number, "
                          "example `r=-5.1` or rectangular region, example "
                          "`[-5, 10.5, 0, 100]`."))
    
    # type of TDS, RDDE and DDAE -> OK, NDDE -> convert to DDAE
    if isinstance(tds, NDDE):
        tds = tds.to_ddae()

	# compress tds (this also sorts)
    tds = tds.compress()

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
    cr, cr_info = roots_ddae(E, A, hA, r)

    return cr, cr_info
