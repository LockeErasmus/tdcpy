"""
NDDE - Neutral Delay Differential Equation
------------------------------------------
TODO TODO TODO
Notes:
    1. As of now, only A and hA, rest of the implementation will follow
"""

import numpy as np
import numpy.typing as npt

from .base import TDSBase
from .ddae import DDAE

class NDDE(TDSBase):
    """ Neutral Delay Differential Equaton

    dxdt(t) = A[i]*x(t - hA[i]) + ... + A[mA]*x(t - hA[mA])
              - H[0] * dxdt(t - hH[0]) - ... - H[mH] * dxdt(t - hH[mH])
    
    TODO documentation

    """
    def __init__(self, A: npt.NDArray, hA: npt.NDArray, H: npt.NDArray, hH: npt.NDArray,
                 B1=None, hB1=None, C1=None, hC1=None, D1=None, hD11=None,
                 B2=None, hB2=None, C2=None, hC2=None, D12=None, hD12=None,
                 D21=None, hD21=None, D22=None, hD22=None) -> None:
        super().__init__()

        # TODO perform checks
        # assert len(A) > 0, "TODO"
        # TODO make sure hA[0] == 0

        # TODO assert nonempty H and non-empty A
        assert np.all(hA >= 0)
        assert np.all(hH > 0) # no 0 delays allowed in hH
        
        ## dynamics
        self._A = A
        self._hA = hA
        self._H = H
        self._hH = hH
        
        # TODO rest of the system description
        # self._B1 , ...

    @property
    def n(self) -> int:
        return self._A[0].shape[0]

    @property
    def E(self) -> npt.NDArray:
        return np.eye(self.n) # TODO this is not correct
    
    @property
    def A(self) -> npt.NDArray:
        """ list of dynamics matrices """
        return self._A
    
    @property
    def H(self) -> npt.NDArray:
        """ list of dynamics matrices """
        return self._H
    
    @property
    def hA(self) -> npt.NDArray:
        """ A delays """
        return self._hA
    
    @property
    def mA(self) -> int:
        """ number of A delays """
        return len(self.hA)
    
    @property
    def hH(self) -> npt.NDArray:
        """ H delays """
        return self._hH
    
    @property
    def mH(self) -> int:
        """ number of H delays """
        return len(self.hH)
    
    @property
    def is_compressed(self) -> bool:
        """ Cheks if NDDE is in compressed form (no duplicates in hA) """
        if len(self.hA) == len(np.unique(self.hA)):
            return True
        else:
            return False
    
    @property
    def is_sorted(self) -> bool:
        """ TODO Checks if DDAE is in sorted form (ascending hA) """
        return all(self.hA[i] <= self.hA[i+1] for i in range(len(self.hA) - 1))
    
    @property
    def is_logical(self) -> bool:
        """ TODO - as of now, always return False """
        return False
        
    @property
    def is_lti(self) -> bool:
        """ Checks if NDDE is Linear Time-invariant """
        return True # as of now, always assume NDDE is LTI
    
    @property
    def is_delay_difference_equation(self) -> bool:
        """ NDDE can not be delay difference equation """
        return False
    
    def sort(self, inplace=False) -> 'NDDE':
        """ Sorts delays (self.hA) into ascending order """
        raise NotImplementedError("Not implemented yet")
    
    def compress(self, inplace=False) -> 'NDDE':
        """ Removes delay duplicates, sorts delays into ascending order
        
        Args:
            inplace (bool): wheter to modify existing RDDE or create a new one,
                default False
        """
        raise NotImplementedError("Not implemented yet")
    
    def to_ddae(self) -> 'DDAE':
        """ Converts NDDE to DDAE
        
        
         [0  I]  [dxdt(t)] = [A0 0] [x(t)] + SUM [A[k]   0] [x(t-tau)]
         [0  0]  [dadt(t)] = [I -I] [a(t)]   k=1 [H[k-1] 0] [a(t-tau)]
        
        
        """
        dtype = self.A.dtype # TODO
        n = self.n
        mA = self.mA
        mH = self.mH
        # construct matrix E
        E = np.zeros(shape=(2*n, 2*n))
        E[:n, n:] = np.eye(n)

        # construct matrix A
        A = np.zeros(shape=(2*n, 2*n, mA+mH))
        A[:n, :n, :mA] = self.A
        A[n:, :n, 0] = np.eye(n) # hA[0] == 0 is assumed
        A[n:, n:, 0] = -np.eye(n) # hA[0] == 0 is assumed
        A[n:, :n, mA:] = self.H

        # construct delays
        hA = np.zeros(shape=(mA+mH,))
        hA[:mA] = self.hA
        hA[mA:] = self.hH

        # TODO construct correct input/output DDAE matrices

        # TODO compress yes or no?

        return DDAE(E=E, A=A, hA=hA)
    
    def get_delay_difference_equation(self) -> DDAE:
        """ Converts to Delay-difference Equation

        For a NDDAE, the associated delay difference equation is given by
            I*x(t) + H[0]*x(t-hH[0]) + ... + H[mH]*x(t-hH[mH]) = 0
        """
        if self.is_logical:
            raise ValueError(f"Can't form Delay-Difference Equation from logical")
        hD = np.concatenate([[0], self.hH], axis=0)
        D = np.concatenate([np.eye(self.n)[:,:,np.newaxis], self.H], axis=2)
        return DDAE(E=np.zeros(shape=(self.n, self.n)), A=D, hA=hD)
