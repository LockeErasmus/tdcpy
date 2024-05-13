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
from .stability.discretization_heuristic import compute_n_rhp, compute_n_rect
from .gamma_r import gamma_r

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

    # CASE 1: ODE or DAE (no delays)
    if hA.shape[0] == 1:
        logger.debug("CASE: ODE or DAE -> finite number of roots")
        # just use eig and filter based on rhp or region
        result = linalg.eig(A[:,:,0], E, left=False, right=False)
        if case == "rhp":
            mask = np.isfinite(result) & (np.real(result)>=r)
            return result[mask]
        else: # case == "region"
            mask = (np.isfinite(result) & (np.real(result)>=r[0]) 
                    & (np.real(result)<=r[1]) & (np.imag(result)>=r[2])
                    & (np.imag(result)<=r[3]))
            return result[mask]

    # CASE 2:
    mA = hA.shape[0] # number of delay terms

    # scale tds such that maximal delay-value equals 1
    # lambda_hat = lambda*tau_m
    # det(lambda E - A0 - A1 *exp(-lambda tau_1) - ... - Am *exp(-lambda tau_m)) = 0
    # => det(lambda_hat E - tau_m *A0 - tau_m * A1 *exp(-lambda_hat tau_1/tau_m) - ... - tau_m Am *exp(-lambda_hat)) = 0
	# re-scaled system matrices and delays are stored in K and tau_s
    tau_max = hA[-1] # last delays is maximal one
    tau_s = hA / tau_max # scale tau vector
    K = tau_max * A # scale matrices of dynamics

    ###########################################################
    # Heuristic for N (degree of the spectral discretisation) #
    ###########################################################    
    # obtain discretization
    discretization = kwargs.get("discretizaton", None)
    if case == "rhp":
        if discretization is None: # envoke heuristic
            # rescale r
            rs = r * tau_max
            # introduce shift of the origin, shifted matrices B, C
            B = K[:,:,0] + (-rs)*E
            C = K[:,:,1:] * np.exp(-rs * tau_s[1:])

            if False: # TODO line 289 - 295, as of now unimportant, later KWARG
                ...
            else:
                diff = tds.to_delay_difference_equation()
                if diff is not None: # empty associated delay difference equation (E is not singular)
                    if diff.hA[0] != 0 or False:
                        raise ValueError("The provided DDAE does not satisfy assumption 2.1.")
                    elif gamma_r(diff, r) >= 1.0:
                        discretization = 30
                        logger.warning((f"Gamma_r exceeds {gamma_r} >= 1 (i.e., CD>r). Spectral "
                                        "discretization with N = 30 (lowered if maximum size of "
                                        "eigenvalue problem is exceeded). Try specifying a "
                                        "rectangular region instead."))
                
                if discretization is None: # discretization is still undefined
                    # region RHP contains finitely many roots
                    discretization = compute_n_rhp(E, B, C, tau=hA) # TODO        
        else: # discretization provided by user -> peform checks
            assert isinstance(discretization, int), "discretization has to be int"
            assert discretization > 1, "discretization has to be > 1"
            logger.debug(f"User provided {discretization=}")

    else: # case == "region":
        if discretization is None: # envoke heuristic
            discretization, origin = compute_n_rect(r, tau_max)
        else:
            assert isinstance(discretization, int), "discretization has to be int"
            assert discretization > 1, "discretization has to be > 1"
            origin = tau_max * ((r[0]+r[1])/2) + 1j*((r[2]+r[3])/2)
            logger.debug(f"User provided {discretization=} | {origin=} ")


    max_size_evp = kwargs.get("max_size_evp", 600)
    assert n <= max_size_evp, "The size of the delay differential equation exceeds max_size_evp"
    discretization_max = np.floor(max_size_evp / n) - 1
    if discretization > discretization_max:
        afm = n*(discretization_max + 1)
        logger.warning(())


    


    
    N_max = np.floor(max_size_evp / n) - 1 # condition: (N+1)*n <= max_size_evp

    # TODO continue here with logic from row 282
    








    
    


