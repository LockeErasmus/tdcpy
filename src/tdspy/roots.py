"""
Implementation of tds_roots
"""
import logging

import numpy as np
import numpy.typing as npt

from .ddae import DDAE
from .stability.discretization_heuristic import compute_n_rhp

logger = logging.getLogger(__name__)

def roots(tds: DDAE, r=0.0, **kwargs):
    """

    Args:
        tds (TODO)
        r (int or list): region, specify r as number on real axis or rectangular
            region via 4 coordinates [Re_min, Re_max, Im_min, Im_max], default r=0.0

    kwargs:
        max_size_evp (int): TODO, default 600
        discretization (int): discretization, if None heuristic is envoked,
            default None, keep default if you don't know, has to be > 1
    """

    # checks for region definition
    if isinstance(r, (int, float)):
        # all OK
        case = "rhp"
    elif isinstance (r, list):
        assert len(r) == 4, "region has to be defined in form [a,b,c,d]"
        assert r[0] < r[1] and r[2] < r[3], "region has to be defined as [a,b,c,d], a<b, c<d"
        # TODO assert all from r finite ?
        case = "rect"
    else:
        raise ValueError(("Region (argument `r`) has to be defined as number, "
                          "example `r=-5.1` or rectangular region, example "
                          "`[-5, 10.5, 0, 100]`."))

	# TODO sort and compress

    # unpack TDS object
    n = tds.n
    E = tds.E

    if tds.mA == 0:
        hA = [0]
        A = [np.zeros(shape=(n,n), dtype=E.dtype)]
    elif tds.hA[0] > 0:
        hA = [0] + tds.hA
        A = [np.zeros(shape=(n,n), dtype=E.dtype)] + tds.A
    else:
        hA = tds.hA
        A = tds.A

    # TODO case 1: ODE





    mA = len(hA) # number of delay terms

    # scale tds such that maximal delay-value equals 1
    # lambda_hat = lambda*tau_m
    # det(lambda E - A0 - A1 *exp(-lambda tau_1) - ... - Am *exp(-lambda tau_m)) = 0
    # => det(lambda_hat E - tau_m *A0 - tau_m * A1 *exp(-lambda_hat tau_1/tau_m) - ... - tau_m Am *exp(-lambda_hat)) = 0
	# re-scaled system matrices and delays are stored in K and tau_s
    tau_max = hA[-1] # last delays is maximal one
    tau_s = [tau/tau_max for tau in hA]
    K = [tau_max * Ai for Ai in A]

    ###########################################################
    # Heuristic for N (degree of the spectral discretisation) #
    ###########################################################    
    # obtain discretization
    discretization = kwargs.get("discretization", None)
    if discretization is None: # envoke heuristic
        # rescale r
        rs = r * tau_max
        # introduce shift of the origin, shifted matrices B, C
        B = K[0] + (-rs)*E
        C = np.zeros(shape=(n,n, mA-1)) # TODO dtype of matrix?
        for i in range(0, mA-1):
            C[:,:, i] = K[i+1] * np.exp(-rs*tau_s[i+1])
        discretization = compute_n_rhp(E, B, C, tau=tds.hA) # TODO

    else: # perform check on user-provided discretization
        assert isinstance(discretization, int), "discretization has to be int"
        assert discretization > 1, "discretization has to be > 1"

    
    
    
    max_size_evp = kwargs.get("max_size_evp", 600)
    assert n <= max_size_evp, "The size of the delay differential equation exceeds max_size_evp"
    N_max = np.floor(max_size_evp / n) - 1
    


    
    N_max = np.floor(max_size_evp / n) - 1 # condition: (N+1)*n <= max_size_evp

    # TODO continue here with logic from row 282
    








    
    


