"""
Implementation of tds_roots
"""
import logging

import numpy as np
import numpy.typing as npt

from .ddae import DDAE

logger = logging.getLogger(__name__)

def roots(tds: DDAE, r=0.0, **kwargs):
    """

    kwargs:
        max_size_evp (int): TODO, default 600
    """
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



    # case 2: DDE
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
    # TODO -> move to separate function
    max_size_evp = kwargs.get("max_size_evp", 600)
    if n <= max_size_evp:
        raise NotImplementedError("The size of the delay differential equation exceeds max_size_evp")
    
    N_max = np.floor(max_size_evp / n) - 1 # condition: (N+1)*n <= max_size_evp

    # case 2.1 - RHP (default case)
    rs = r * tau_max # rescale r
    # introduce a shift of the origin, shifted matrices are stored in B and C


    C = np.zeros(shape=(n,n, mA-1)) # TODO dtype
    for i in range(0, mA-1):
        C[:,:, i] = K[i+1] * np.exp(-rs*tau_s[i+1])

    # TODO continue here with logic from row 282
    








    
    


