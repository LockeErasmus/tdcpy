"""
RDDE - Retarded Delay Differential Equation
-------------------------------------------

Notes:
    1. As of now, only A and hA, rest of the implementation will follow
"""

import numpy as np
import numpy.typing as npt

from .base import TDSBase
from .ddae import DDAE

class RDDE(TDSBase):
    """ Retarded Delay Differential Equaton
    
    TODO documentation

    """
    def __init__(self, A: npt.NDArray, hA: npt.NDArray,
                 B: npt.NDArray=None, hB: npt.NDArray=None, C: npt.NDArray=None, hC: npt.NDArray=None, 
                 D: npt.NDArray=None, hD: npt.NDArray=None, **kwargs) -> None:
        A, hA = self._prepare_system_descriptor_matrix_vector(A, hA, allow_empty=False,
                                                              allow_negative_delays=False,
                                                              allow_complex=False,
                                                              add_zero_delay=True,
                                                              sort_by_delays=True,
                                                              dtype=kwargs.get("dtype", np.float64))

        # Input/Output matrices
        if B is not None or hB is not None:
            B, hB = self._prepare_system_descriptor_matrix_vector(B, hB,
                                                                  allow_empty=True,
                                                                  allow_negative_delays=False,
                                                                  allow_complex=False,
                                                                  add_zero_delay=False,
                                                                  sort_by_delays=True,
                                                                  dtype=kwargs.get("dtype", np.float64))
            if B.shape[0] != A.shape[0]:
                raise ValueError("B shape does not match A shape")
        
        if C is not None or hC is not None:
            C, hC = self._prepare_system_descriptor_matrix_vector(C, hC,
                                                                  allow_empty=True,
                                                                  allow_negative_delays=False,
                                                                  allow_complex=False,
                                                                  add_zero_delay=False,
                                                                  sort_by_delays=True,
                                                                  dtype=kwargs.get("dtype", np.float64))
            if C.shape[1] != A.shape[1]:
                raise ValueError("C shape does not match A shape")
        
        if D is not None or hD is not None:
            if C is None or hC is None:
                raise ValueError("C, hC needs to be defined to define D, hD")
            D, hD = self._prepare_system_descriptor_matrix_vector(D, hD,
                                                                  allow_empty=True,
                                                                  allow_negative_delays=False,
                                                                  allow_complex=False,
                                                                  add_zero_delay=False,
                                                                  sort_by_delays=True,
                                                                  dtype=kwargs.get("dtype", np.float64))
            if D.shape[0] != C.shape[0]:
                raise ValueError("D shape does not match C shape")        

        # set all matrices and delays as class attributes
        self._A = A
        self._hA = hA
        self._B = B
        self._hB = hB
        self._C = C
        self._hC = hC
        self._D = D
        self._hD = hD

    @property
    def n(self) -> int:
        return self._A[0].shape[0]
    
    @property
    def n_inputs(self) -> int:
        """ number of inputs """
        if self.B is None:
            return 0
        return self.B.shape[1]

    @property
    def n_outputs(self) -> int:
        """ number of outputs """
        if self.C is None:
            return 0
        return self.C.shape[0]

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
    def B(self) -> npt.NDArray | None:
        """ input matrices """
        return self._B
    
    @property
    def hB(self) -> npt.NDArray | None:
        """ input delays """
        return self._hB
    
    @property
    def mB(self) -> int:
        """ number of input delays """
        if self._hB is None:
            return 0
        return self.hB.shape[0]

    @property
    def C(self) -> npt.NDArray | None:
        """ output matrices """
        return self._C
    
    @property
    def hC(self) -> npt.NDArray | None:
        """ output delays """
        return self._hC
    
    @property
    def mC(self) -> int:
        """ number of output delays """
        if self._hC is None:
            return 0
        return self.hC.shape[0]

    @property
    def D(self) -> npt.NDArray | None:
        """ feed-through matrices """
        return self._D
    
    @property
    def hD(self) -> npt.NDArray | None:
        """ feed-through delays """
        return self._hD
    
    @property
    def mD(self) -> int:
        """ number of feed-through delays """
        if self._hD is None:
            return 0
        return self.hD.shape[0]
    
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
        """ RDDE can NOT be delay difference equation """
        return False
    
    def sort(self, inplace=False) -> 'RDDE':
        """ Sorts delays (self.hA) into ascending order """
        raise NotImplementedError("Not implemented yet")
    
    def compress(self, inplace=False, rtol=1e-5, atol=1e-8) -> 'RDDE':
        """ Removes delay duplicates, sorts delays into ascending order
        
        Args:
            inplace (bool): wheter to modify existing DDEA or create a new one,
                default False
            rtol (float): relative tolerance for determining matrix element is
                zero, default 1e-5
            atol (float): absolute tolerance for determining matrix element is 
                zero, default 1e-8
        
        Returns:
            RDDE: compressed representation
            None: if inplace=True (current object is updated)
        """
        unique_hA = np.unique(self.hA) # sorted in ascending order
        newA = np.zeros(shape=(self.n, self.n, unique_hA.shape[0]))
        for i in range(unique_hA.shape[0]):
            mask = self.hA == unique_hA[i] # create mask
            newA[:,:,i] = np.sum(self.A[:,:,mask], axis=2)
        
        # perform elimination of newAi close to 0.0
        mask = np.all(np.isclose(newA, 0.0, rtol=rtol, atol=atol), axis=(0,1))
        A = newA[:,:,~mask]
        hA = unique_hA[~mask]

        # TODO also solve input matrices, output matrices
        
        if inplace:
            self._A = A
            self._hA = hA
        else:
            return RDDE(A=A, hA=hA)
    
    def get_delay_difference_equation(self) -> None:
        """ RDDE has no associated delay difference equation """
        return None
