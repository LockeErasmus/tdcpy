"""
RDDE - Retarded Delay Differential Equation
-------------------------------------------

Notes:
    1. As of now, only A and hA, rest of the implementation will follow
"""
import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .common.compress import compress_matrices_delays, sort_matrices_delays
from .base import TDSBase

logger = logging.getLogger(__name__)


class RDDE(TDSBase):
    """ Retarded Delay Differential Equaton
    
    TODO documentation

    """
    def __init__(self, A: npt.NDArray, hA: npt.NDArray, 
                 B: npt.NDArray=None, hB: npt.NDArray=None, C: npt.NDArray=None, hC: npt.NDArray=None, 
                 D: npt.NDArray=None, hD: npt.NDArray=None, **kwargs) -> None:
        """
        
        Args: 


            **kwargs:
                dtype (): default np.float64
                tol_singular (float): considering value singular, default 1e-12

        
        """
        # A, hA
        assert isinstance(A, np.ndarray) and isinstance(hA, np.ndarray), "both ndarrays"
        assert A.ndim  == 3 and hA.ndim == 1, "dimensions check 1"
        assert A.shape[2] == hA.shape[0], "number of delays hA does not match number of matrices Ai"
        assert np.all(hA >= 0.0), "only non-negative delays possible"
        if not np.any(hA == 0):
            hA = np.r_[0.0, hA]
            A = np.concatenate([np.zeros((A.shape[0],A.shape[1], 1), dtype=A.dtype), A], axis = 2)

        # I/O matrices
        if B is not None or hB is not None:
            # input matrices are defined
            assert isinstance(B, np.ndarray) and isinstance(hB, np.ndarray), "both ndarrays"
            assert B.ndim == 3 and hB.ndim == 1, "dimensions check 1"
            assert B.shape[0] == A.shape[0], "shapes of system do not match A-B matrices"
            assert B.shape[2] == hB.shape[0], "number of delays hB do not match number of matrices Bi"
            assert np.all(hB >= 0.0), "only non-negative delays possible"

        if C is not None or hC is not None:
            # output matrices are defined
            assert isinstance(C, np.ndarray) and isinstance(hC, np.ndarray), "both ndarrays"
            assert C.ndim == 3 and hC.ndim == 1, "dimensions check 1"
            assert C.shape[1] == A.shape[1], "shapes of system does not match A-C matrices "
            assert C.shape[2] == hC.shape[0], "number of delays hB does not match number of matrices Bi"
            assert np.all(hC >= 0.0), "only non-negative delays possible"
        
        if D is not None or hD is not None:
            # feed-through matrices are defined
            assert isinstance(C, np.ndarray), "C, hC needs to be defined to define D, hD"
            assert isinstance(D, np.ndarray) and isinstance(hD, np.ndarray), "both ndarrays"
            assert D.ndim == 3 and hD.ndim == 1, "dimensions check 1"
            assert D.shape[0] == C.shape[0], "shapes of system does not match C-D matrices "
            assert D.shape[2] == hD.shape[0], "number of delays hB does not match number of matrices Bi"
            assert np.all(hD >= 0.0), "only non-negative delays possible"

        # --- ARGS ---
        ## dynamics
        self._A = A
        self._hA = hA
        self._B = B
        self._hB = hB
        self._C = C
        self._hC = hC
        self._D = D
        self._hD = hD
        
        # TODO checks when for example A, B, C defined and D is not        

        # --- KWARGS ---
        self.dtype = kwargs.get("dtype", np.float64)
        self.tol_singular = kwargs.get("tol_singular", 1e-12)

        super().__init__()

        # TODO perform checks        
        # TODO rest of the system description

    @property
    def n(self) -> int:
        """ number of variables """
        return self.A.shape[1]

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
        return np.eye(self.n, dtype=self.dtype)
        
    @property
    def A(self) -> list[npt.NDArray]:
        """ dynamics matrices """
        return self._A
    
    @property
    def hA(self) -> list[float]:
        """ dynamic delays """
        return self._hA
    
    @property
    def mA(self) -> int:
        """ number of delays """
        return self.hA.shape[0]
    
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
    def uE(self) -> npt.NDArray:
        """ orthonormal basis for left null space of E """
        return np.zeros(shape=(self.n, 0))
    
    @property
    def vE(self) -> npt.NDArray:
        """ orthonormal basis for right null space of E """
        return np.zeros(shape=(self.n, 0))

    @property
    def is_logical(self) -> bool:
        """ property from original MATLAB package, unused as of now """
        return False # as of now, always assume DDAE is not logical
    
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
    def is_real(self) -> bool:
        """ Checks if DDAE uses complex storage for any of defining arrays """
        fields_to_check = ["_A", "_hA", "_B", "_hB",
                           "_C", "_hC", "_D", "_hD"]
        for field in fields_to_check:
            val = getattr(self, field, None)
            if val is not None:
                if not np.isrealobj(val):
                    return False
        return True

    @property
    def is_delay_difference_equation(self) -> bool:
        """ RDDE can NOT be delay difference equation """
        return False
    
    def to_asymptotic_transfer_function(self, **kwargs) -> 'RDDE':
        raise NotImplementedError("Not implemented yet")
    
    def sort(self, inplace=False) -> 'RDDE':
        """ Sorts arrays containing matrices and delays (ascending order) """
        
        A, hA = sort_matrices_delays(self.A, self.hA)

        if self.B is None or self.hB is None:
            B, hB = self.B, self.hB
        else:
            B, hB = sort_matrices_delays(self.B, self.hB)
        
        if self.C is None or self.hC is None:
            C, hC = self.C, self.hC
        else:
            C, hC = sort_matrices_delays(self.C, self.hC)
        
        if self.D is None or self.hD is None:
            D, hD = self.D, self.hD
        else:
            D, hD = sort_matrices_delays(self.D, self.hD)

        if inplace:
            self._A = A
            self._hA = hA
            self._B = B
            self._hB = hB
            self._C = C
            self._hC = hC
            self._D = D
            self._hD = hD
        else:
            return RDDE(A=A, hA=hA, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)
    
    
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
        unique_hA, unique_hB = np.unique(self.hA), np.unique(self.hB) # sorted in ascending order
        unique_hC, unique_hD = np.unique(self.hC), np.unique(self.hD) # sorted in ascending order
        
        newA, newB = np.zeros(shape=(self.n, self.n, unique_hA.shape[0])), np.zeros(shape=(self.B.shape[0], self.B.shape[1], unique_hB.shape[0]))
        newC, newD = np.zeros(shape=(self.C.shape[0], self.C.shape[1], unique_hC.shape[0])), np.zeros(shape=(self.D.shape[0], self.D.shape[1], unique_hD.shape[0]))

        for i in range(unique_hA.shape[0]):
            mask = self.hA == unique_hA[i] # create mask
            newA[:,:,i] = np.sum(self.A[:,:,mask], axis=2)

        # perform elimination of newAi close to 0.0
        mask = np.all(np.isclose(newA, 0.0, rtol=rtol, atol=atol), axis=(0,1))
        A = newA[:,:,~mask]
        hA = unique_hA[~mask]


        # TODO also solve input matrices, output matrices
        for i in range(unique_hB.shape[0]):
            mask = self.hB == unique_hB[i] # create mask
            newB[:,:,i] = np.sum(self.B[:,:,mask], axis=2)

        # perform elimination of newBi close to 0.0
        mask = np.all(np.isclose(newB, 0.0, rtol=rtol, atol=atol), axis=(0,1))
        B = newB[:,:,~mask]
        hB = unique_hB[~mask]

        for i in range(unique_hC.shape[0]):
            mask = self.hC == unique_hC[i] # create mask
            newC[:,:,i] = np.sum(self.C[:,:,mask], axis=2)

        # perform elimination of newCi close to 0.0
        mask = np.all(np.isclose(newC, 0.0, rtol=rtol, atol=atol), axis=(0,1))
        C = newC[:,:,~mask]
        hC = unique_hC[~mask]

        for i in range(unique_hD.shape[0]):
            mask = self.hD == unique_hD[i] # create mask
            newD[:,:,i] = np.sum(self.D[:,:,mask], axis=2)

        # perform elimination of newDi close to 0.0
        mask = np.all(np.isclose(newD, 0.0, rtol=rtol, atol=atol), axis=(0,1))
        D = newD[:,:,~mask]
        hD = unique_hD[~mask]


        if inplace:
            self._A = A
            self._hA = hA
            self._B = B
            self._hB = hB
            self._C = C
            self._hC = hC
            self._D = D
            self._hD = hD
        else:
            return RDDE(A=A, hA=hA, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)
    
    def get_delay_difference_equation(self) -> None:
        """ RDDE has no associated delay difference equation """
        return None
    
    def eval_char_matrix(self, s: complex) -> npt.NDArray:
        """ Evaluate characteristic matrix at `s`

        M(s) = I*s - A[0]*exp(-s*hA[0]) - ... - A[mA]*exp(-s*hA[mA])
        
        Args:
            s (complex, float, int): s from complex plane
        
        Returns:
            M (array): characteristic matrix M evaluated at s
        """
        return self.E*s - np.sum(self.A * np.exp(-s*self.hA), axis=2)

    def eval_char_matrix_derivative(self, s: complex) -> npt.NDArray:
        """ Derivative of characteristic matrix with respect to s evaluated at s

        dM(s) = I + hA[0]*A[0]*exp(-s*hA[0]) + ... + hA[mA]*A[mA]*exp(-s*hA[mA])
        
        Args:
            s (complex, float, int): s from complex plane
        
        Returns:
            dM (array): derivative of characteristic matrix M evaluated at s
        """
        return self.E + np.sum(self.A * self.hA * np.exp(-s*self.hA), axis=2)

