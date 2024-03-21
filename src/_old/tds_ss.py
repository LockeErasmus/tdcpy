import logging
import warnings


from typing import Type
import numpy as np
import numpy.typing as npt
import numpy.linalg as nplinalg

from scipy import linalg

from .tds_base import TimeDelaySystem

logger = logging.getLogger(__name__)

class TDS(TimeDelaySystem):
    """ State-space representation for LTI time-delay systems """


    def __init__(self, E: npt.NDArray, A=list[npt.NDArray], hA=list[float]) -> None:
        
        assert len(A) > 0, f"At least one A matrix is expected"
        assert len(A) == len(hA), "Same number of A and delays"
        # TODO assert all elements hA are unique

        # sort delays and matrices of dynamics (delays ascending order)
        sorted_pairs = sorted(zip(A, hA))
        self._A, self._hA = [list(t) for t in zip(*sorted_pairs)]
        self._E = E

        # TODO compress delays - no duplicates

        # TODO add A0=zeros() and hA = [0, hA] if there is no 0.0 delay
    
    @property
    def n(self):
        """ number of delay terms """
        return len(self._hA) - 1
    
    @property
    def hA(self):
        """ delays """
        return self._hA
    
    @property
    def A(self):
        """ dynamic matrices """
        return self._A

def tds_gamma_r(tds: Type[TimeDelaySystem], r: float|int, **kwargs):
    """ """

    if tds.n < 2:
        gamma_r = 0.0

    # assume tds is in delay difference form

def tds_roots(tds: TDS, r: float, **kwargs):
    """
    
    source: https://gitlab.kuleuven.be/u0011378/tds-control/-/blob/main/tds-control/code/tds_roots.m
    """
    info = {
        "max_size_evp_enforced": False,
    }

    # TODO compress
    # TODO sort delays

    # unpack values
    n = len(tds._hA)# number of delays
    E = tds._E
    A = tds._A
    hA = tds._hA

    # CASE 1 : TODO - no delay --> ODE

    # CASE 2 : TODO -> DDE
    mA = len(hA) # number of delays

    ## re-scale that max(hA) == 1.0
    tau_s = [d/hA[-1] for d in hA]
    K = []
    for a in A:
        K.append(hA[-1]*a)
    


    # heuristics for N (degree of spectral discretization)
    fix_N = kwargs.get("fix_N", None)
    if fix_N:
        assert isinstance(fix_N, int) and fix_N > 0, "kwarg `fix_N` has to be integer greater than 0"
    max_size_evp =  kwargs.get("max_size_evp", 600)
    N_max = int(max_size_evp / n) - 1 # condition: (N+1)*n <= max_size_evp

    if type(r) in [float, int]: # region is specified by number on real axis
        if fix_N: # user fixed N
            N = fix_N
        else: # N is calculated by heuristic and capped by max size
            rs = r * hA[-1] # rescale r
            # introduce a shift of the origin, shifted matrices are stored in B and C
            B = [k + -rs*E for k in K]
            C = np.zeros(shape=(n, n, mA-1))
            for i in range(mA-1):
                C[:,:,i] = K[i+1]*np.exp((-rs) * tau_s[i+1])
            
            # automatically determine necessary degree to capture all cr in the desired rhp
            tds_ = tds

            if tds.n != 0:
                if tds.hA[0] != 0.0 or nplinalg.matrix_rank(tds.A[0]) < len(tds.A[0]): # TODO len
                    raise ValueError("The provided DDAE does not satisfy Assumption 2.1.")
                else:
                    pass
                # TODO calculate N
                N = 750
            
            if N > N_max:
                N = N_max
                afm = n * (N_max + 1)
                warnings.warn(
                    (f"Size of the generalized EVP would exceed its maximum value. Discretization around {max(0, r)} + 0.0j"
                    f" with N = {N_max} instead (size of new eigenvalue problem: {afm} x {afm}). As a consequence, not all"
                    f"characteristic roots in the\nspecified right half-plane might be found. To make sure that all "
                    f"desired characteristic roots are found either increase r (i.e., shift the desired right half-plane "
                    f"to the right) or the kwarg `max_size_evp`.")
                )
                info["max_size_evp_enforced"] = True

        QQ = A
        if info.get("max_size_evp_enforced") and r < 0:
            # Case: N is lowered and r < 0, use rs = 0 (ie. no shift)
            rs = 0
            QQ = K
        else:
            # Case: Use shift rs
            QQ[0] = B
            for i in range(1, mA):
                QQ[i] = C[:,:,i-1]
    elif isinstance(r, tuple, list): # []
        assert len(r) == 4, "Length of defined `r` (region) has to have 4 members "
        # TODO assert region is well defined

        if fix_N:
            N=fix_N
            origin = hA[mA] * ((r[1] + r[2]) / 2.) + 1j * ((r[3] + r[4]) / 2.)
        else: # N is calculated by heuristic and capped by max size
            pass # TODO
    else:
        raise ValueError("Only r=number or r=region [a,b,c,d] is supported")
    
    logger.debug(f"Degree of spectral discretization: N = {N}")
    info["N"] = N

    


        




if __name__ == "__main__":
    pass





