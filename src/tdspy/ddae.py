"""
DDAE implementation
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg


logger = logging.getLogger(__name__)


class DDAE:

    def __init__(self, A: list[npt.NDArray], hA: list[float], E=None, uE=None, vE=None,**kwargs) -> None:
        
        assert len(A) >= 1, "At least one matrix of dynamics is required"
        assert len(A) == len(hA)
        assert all(delay>=0 for delay in hA), "Only positive delays possible"
        
        
        assert np.atleast_2d(*A)
        shape = A[0].shape
        assert all([a.shape == shape for a in A]), "A_i Matrices have to have same shape"
        assert all([a.shape[0] == a.shape[1] for a in A]), "A_i matrices have to be square"

        # TODO if necessary, add 0 delay term
        
        # ---
        self._A = A
        self._hA = hA

        self._E = E
        self._uE = uE
        self._vE = vE

        self._n = shape[0]

        # --- KWARGS
        self.dtype = kwargs.get("dtype", np.float64)
        self.tol_singular = kwargs.get("tol_singular", 1e-12)

    @property
    def n(self) -> int:
        return self._n

    @property
    def E(self) -> npt.NDArray:
        if self._E is None:
            return np.eye(self.n, dtype=self.dtype)
        else:
            return self._E
    
    @property
    def A(self) -> list[npt.NDArray]:
        """ list of dynamics matrices """
        return self._A
    
    @property
    def hA(self) -> list[float]:
        """ delays """
        return self._hA
    
    @property
    def mA(self) -> int:
        """ number of delays """
        return len(self.hA)

    @property
    def uE(self) -> npt.NDArray:
        """ orthonormal basis for left null space of E """
        if self._uE is None:
            uE = linalg.null_space(self.E.T, rcond=self.tol_singular)
            return uE
        else:
            return self._uE
    
    @property
    def vE(self) -> npt.NDArray:
        """ orthonormal basis for right null space of E """
        if self._vE is None:
            vE = linalg.null_space(self.E, rcond=self.tol_singular)
            return vE
        else:
            return self._vE
        
    @property
    def is_logical(self) -> bool:
        """ property from original MATLAB package, unused as of now """
        return False # as of now, always assume DDAE is not logical
    
    @property
    def is_compressed(self) -> bool:
        """ Cheks if DDAE is in compressed form (no duplicates in hA) """
        if len(self.hA) == len(np.unique(self.hA)):
            return True
        else:
            return False
    
    @property
    def is_sorted(self) -> bool:
        """ Checks if DDAe is in sorted form (ascending hA) """
        return all(self.hA[i] <= self.hA[i+1] for i in range(len(self.hA) - 1))
    
    @property
    def is_lti(self) -> bool:
        """ Checks if DDAE is Linear Time-invariant """
        return True # as of now, always assume DDAE is LTI
    
    @property
    def is_real(self) -> bool:
        """ Checks if DDAE uses complex storage for any of defining matrices """
        raise NotImplementedError()
    
    @property
    def is_delay_difference_equation(self) -> bool:
        """ Checks if DDAE is a delay difference equation, i.e. E==0 """
        return not self.is_logical and np.allclose(self.E, 0, atol=1e-12)

    def to_delay_difference_equation(self, **kwargs) -> 'DDAE':
        """ Converts to Delay-difference Equation 

        For a DDAE, the associated delay difference equation is given by
            U'*A[0]*V x(t-hA(1)) + ... + U'*A[mA]*V x(t-hA(mA)) = 0
        with U and V orthogonal matrices whose columns form a basis for the null
        space of E.
        
        kwargs:
            tol (float): norm tolerance for considering matrix vanish,'
                default 1e-14

        """
        if self.is_logical:
            raise ValueError(f"Can't form DDE from logical")
        
        tol = kwargs.get("tol", 1e-14)
                
        D = []
        hD = []

        uE = self.uE
        vE = self.vE

        #if np.size(uE) == 0:
        # TODO case

        norm_uE = linalg.norm(uE, ord=1, axis=None)
        norm_vE = linalg.norm(vE, ord=1, axis=None)
        
        for Ai, hAi in zip(self.A, self.hA):
            Di = np.transpose(uE) @ Ai @ vE
            norm_mat = linalg.norm(Di, ord=1, axis=None)
            if norm_mat / max(norm_uE, norm_vE) > tol:
                D.append(Di)
                hD.append(hAi)
        
        nE = uE.shape[1] # TODO WIP
        dtype = self.E.dtype # numpy data type
        diff = DDAE(A=D, hA=hD, E=np.zeros(shape=(nE,nE), dtype=dtype),
                    uE=np.eye(nE, dtype=dtype), vE=np.eye(nE, dtype=dtype))
        return diff

    def to_asymptotic_transfer_function(self, **kwargs) -> 'DDAE':
        raise NotImplementedError("Not implemented yet")
    
    def sort(self, inplace=False):
        """ Sorts delays (mainly hA) into ascending order """
        raise NotImplementedError("Not implemented yet")
    
    def compress(self, inplace=False):
        """ Removes delay duplicates (matrices are added) """
        raise NotImplementedError("Not implemented yet")
    

def normalize_delay_difference_equation(diff: DDAE):
    """ normalizes delay difference equation

    Transforms the delay difference equation such that the leading zero delay
    matrix A0 equals identity.

    Args:
        diff (DDAE): DDAE in form of delay difference equation (E=0)

    Returns:
        tuple containing

        - D (list of array): list of matrices DD = [inv(A0)*A1, ... , inv(A0)*Am]
        - hDD (array): array of non-zero delays
    """
    hDD = diff.hA[1:] # TODO assume at least 2 delays, i.e. [0, tau1]

    if diff.mA == 2:
        DD = [linalg.lstsq(diff.A[0], diff.A[1])]
        return DD, hDD
    
    P, L, U = linalg.lu(diff.A[0]) # LU decomposition for having inverse of A0
    DD = []
    for i in range(1, diff.mA):
        DD.append(
            linalg.lstsq(U, linalg.lstsq(L, P @ diff.A[i])) # Di = inv(A0) @ Ai
        )
    return DD, hDD



