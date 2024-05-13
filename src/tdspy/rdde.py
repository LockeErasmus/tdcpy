"""
RDDE - Retarded Delay Differential Equation
-------------------------------------------

Notes:
    1. As of now, only A and hA, rest of the implementation will follow
"""

import numpy as np
import numpy.typing as npt

from .base import TDSBase

class RDDE(TDSBase):
    """ Retarded Delay Differential Equaton
    
    TODO documentation

    """
    def __init__(self, A, hA,
                 B1=None, hB1=None, C1=None, hC1=None, D1=None, hD11=None,
                 B2=None, hB2=None, C2=None, hC2=None, D12=None, hD12=None, D21=None, hD21=None, D22=None, hD22=None) -> None:
        super().__init__()

        # TODO perform checks
        assert len(A) > 0, "TODO"

        ## dynamics
        self._A = A
        self._hA = hA
        
        # TODO rest of the system description

    @property
    def n(self) -> int:
        return self._A[0].shape[0]

    @property
    def E(self) -> npt.NDArray:
        return np.eye(self.n)
    
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
    def p2(self) -> int:
        """ number of delays """
        raise NotImplementedError("p2 RDDE")
    
    @property
    def is_compressed(self) -> bool:
        """ Cheks if RDDE is in compressed form (no duplicates in hA) """
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
        return True # as of now, always assume RDDE is LTI
    
    @property
    def is_delay_difference_equation(self) -> bool:
        """ RDDE can not be delay difference equation """
        return False
    
    def sort(self, inplace=False) -> 'RDDE':
        """ Sorts delays (self.hA) into ascending order """
        raise NotImplementedError("Not implemented yet")
    
    def compress(self, inplace=False) -> 'RDDE':
        """ Removes delay duplicates, sorts delays into ascending order
        
        Args:
            inplace (bool): wheter to modify existing RDDE or create a new one,
                default False
        """
        raise NotImplementedError("Not implemented yet")


