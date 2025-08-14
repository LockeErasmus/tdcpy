"""
DDAE implementation


TODO:
    1. `E` should not have default None value and should be first arg?
    1. lot of checking is duplicated code, maybe function(s)? but then we lose
        flexibility, as sometimes you need to add special check ...
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .common.compress import compress_matrices_delays, sort_matrices_delays
from .common.delay_difference_equation import ddae_to_diff
from .base import TDSBase

logger = logging.getLogger(__name__)


class DDAE(TDSBase):
    """
    """

    def __init__(self, A: npt.NDArray, hA: npt.NDArray, E: npt.NDArray=None, uE: npt.NDArray=None, vE: npt.NDArray=None,
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
        assert A.ndim == 3 and hA.ndim == 1, "dimensions check 1"
        assert A.shape[2] == hA.shape[0], "number of delays  hA does not match number of matrices Ai"
        assert np.all(hA >= 0.0), "only non-negative delays possible"
        if not np.any(hA == 0): # if necessary, add 0 delay term
            hA = np.r_[0.0, hA]
            A = np.concatenate([np.zeros((A.shape[0],A.shape[1], 1), dtype=A.dtype), A], axis=2)
        
        # E, uE, vE
        if E is not None:
            assert isinstance(E, np.ndarray)
            assert E.ndim == 2
            assert E.shape[0] == A.shape[0] and E.shape[1] == A.shape[1], f"{E.shape=}, {A.shape=}"
        if uE is not None:
            assert isinstance(uE, np.ndarray)
            if uE.size > 0:
                assert uE.ndim == 2, "ndim of nullspace has to be 2"
                assert uE.shape[0] == A.shape[0], "uE^T @ Ai has to be possible (dimensions has to match)"
        if vE is not None:
            assert isinstance(vE, np.ndarray)
            if vE.size > 0:
                assert vE.ndim == 2, "ndim of nullspace has to be 2"
                assert vE.shape[0] == A.shape[1], "Ai @ vE has to be possible (dimensions has to match)"

        # I/O matrices
        # TODO: tests are (somewhat) repeating, consider function?
        if B is not None or hB is not None:
            # input matrices are defined
            assert isinstance(B, np.ndarray) and isinstance(hB, np.ndarray), "both ndarrays"
            assert B.ndim == 3 and hB.ndim == 1, "dimensions check 1"
            assert B.shape[0] == A.shape[0], "shapes of system does not match A-B matrices"
            assert B.shape[2] == hB.shape[0], "number of delays hB does not match number of matrices Bi"
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
        self._E = E
        self._uE = uE
        self._vE = vE
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
        if self._E is None:
            return np.eye(self.n, dtype=self.dtype)
        else:
            return self._E
    
    @property
    def A(self) -> npt.NDArray:
        """ dynamics matrices """
        return self._A
    
    @property
    def hA(self) -> npt.NDArray:
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
        # TODO - this is not correct as A[:,:,i] can be close to 0 -> not compressed
        if self.mA == len(np.unique(self.hA)):
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
        """ Checks if DDAE uses complex storage for any of defining arrays """
        fields_to_check = ["_E", "_uE", "_vE", "_A", "_hA", "_B", "_hB",
                           "_C", "_hC", "_D", "_hD"]
        for field in fields_to_check:
            val = getattr(self, field, None)
            if val is not None:
                if not np.isrealobj(val):
                    return False
        return True
    
    @property
    def is_delay_difference_equation(self) -> bool:
        """ Checks if DDAE is a delay difference equation, i.e. E==0 """
        return not self.is_logical and np.allclose(self.E, 0, atol=1e-12)
    
    @property
    def is_normalized_delay_difference_equation(self) -> bool:
        """ Checks if DDAE is a normalized delay difference equation
        
        This condition is defined as fullfilling
            (a) DDAE is not logical
            (b) E == 0
            (c) A[0] == I
        with `np.allclose` absolute tolerance 1e-12.
        """
        flag = (not self.is_logical and np.allclose(self.E, 0, atol=1e-12)
                and np.allclose(self.A[:,:,0], np.eye(self.n), atol=1e-12))
        return flag
    
    @property
    def is_essentially_retarded(self) -> bool:
        """ Checks if DDAE is essentially retarded
        
        DDAE is essentially retarded IFF characteristic equation of underlying
        delay-difference equation 
        
            det | SUM A[i] * exp(-s*hA[i]) |
        
        does not depend on complex argument `s`, i.e. delay difference equation
        has to take form:

            0 = A[0] * x(t)
        """
        D, hD = ddae_to_diff(self.E, self.A, self.hA, uE=self.uE, vE=self.vE,
                             tol=1e-12, rcond=1e-14)
        D, hD = compress_matrices_delays(D, hD) # compress matrices
        if np.size(hD) == 0: # empty delay difference equation
            return True
        if hD.shape[0] == 1 and hD[0] == 0.0:
            # delay difference equation is of a form
            # 0 = D[0] * x(t)
            return True
        return False # delay difference equation depends on complex argument

    @property
    def is_essentially_neutral(self):
        """ Checks if DDAE is essentialy netural """
        return not self.is_essentially_retarded
   
    def get_delay_difference_equation(self, **kwargs) -> 'DDAE':
        """ Converts to Delay-difference Equation 

        For a DDAE, the associated delay difference equation is given by
            U'*A[0]*V x(t-hA[0]) + ... + U'*A[mA-1]*V x(t-hA[mA-1]) = 0
        with U and V orthogonal matrices whose columns form a basis for the null
        space of E.
        
        kwargs:
            tol: norm tolerance for considering matrix vanish, default 1e-14
            rcond (float): relative condition number. Singular values s smaller
                than rcond * max(s) are considered zero in null space
                construction, default 1e-12
        
        Returns:
            DDAE representing delay difference equation if E is singular
            None if E is non-singular (there is no delay difference equation)

        """
        if self.is_logical:
            raise ValueError(f"Can't form delay difference equation from logical DDAE")
        
        D, hD = ddae_to_diff(self.E, self.A, self.hA, **kwargs)

        if np.size(D) == 0:
            # E is singular -> empty delay difference equation
            return None # TODO - not good implementation
        
        nE = D.shape[1] # TODO --- what if D.shape[1] == 0 ? can it happen?
        dtype = self.E.dtype
        diff = DDAE(A=D, hA=hD, E=np.zeros(shape=(nE,nE), dtype=dtype),
                    uE=np.eye(nE, dtype=dtype), vE=np.eye(nE, dtype=dtype))
        return diff

    def to_asymptotic_transfer_function(self, **kwargs) -> 'DDAE':
        """ TODO - implement this function """
        raise NotImplementedError("Not implemented") 
    
    def sort(self, inplace=False) -> 'DDAE':
        """ Sorts arrays containing matrices and delays (ascending order)

        Args:
            inplace (bool): wheter to modify existing DDEA or create a new one,
                default False
        
        Returns:
            DDAE: sorted representation
            None: if inplace=True (current object is updated)
        """
        A, hA = None, None
        if self.A is not None and self.hA is not None:
            A, hA = sort_matrices_delays(self.A, self.hA)

        B, hB = None, None
        if self.B is not None and self.hB is not None:
            B, hB = sort_matrices_delays(self.B, self.hB)
        
        C, hC = None, None
        if self.C is not None and self.hC is not None:
            C, hC = sort_matrices_delays(self.C, self.hC)
        
        D, hD = None, None
        if self.D is not None and self.hD is not None:
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
            return DDAE(A=A, hA=hA, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)

    def compress(self, inplace=False, rtol=1e-5, atol=1e-8) -> 'DDAE':
        """ Removes delay duplicates, sorts delays into ascending order
        
        Args:
            inplace (bool): wheter to modify existing DDEA or create a new one,
                default False
            rtol (float): relative tolerance for determining matrix element is
                zero, default 1e-5
            atol (float): absolute tolerance for determining matrix element is 
                zero, default 1e-8
        
        Returns:
            DDAE: compressed representation
            None: if inplace=True (current object is updated)
        """
        A, hA = None, None
        if self.A is not None and self.hA is not None:
            A, hA = compress_matrices_delays(self.A, self.hA)

        B, hB = None, None
        if self.B is not None and self.hB is not None:
            B, hB = compress_matrices_delays(self.B, self.hB)
        
        C, hC = None, None
        if self.C is not None and self.hC is not None:
            C, hC = compress_matrices_delays(self.C, self.hC)
        
        D, hD = None, None
        if self.D is not None and self.hD is not None:
            D, hD = compress_matrices_delays(self.D, self.hD)      

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
            return DDAE(E=self.E, A=A, hA=hA, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)
        
    def eval_char_matrix(self, s: complex) -> npt.NDArray:
        """ Evaluate characteristic matrix at `s`

        M(s) = E*s - A[0]*exp(-s*hA[0]) - ... - A[mA]*exp(-s*hA[mA])
        
        Args:
            s (complex, float, int): s from complex plane
        
        Returns:
            M (array): characteristic matrix M evaluated at s
        """
        return self.E*s - np.sum(self.A * np.exp(-s*self.hA), axis=2)
    
    def eval_char_matrix_derivative(self, s: complex) -> npt.NDArray:
        """ Derivative of characteristic matrix with respect to s evaluated at s

        dM(s) = E + hA[0]*A[0]*exp(-s*hA[0]) + ... + hA[mA]*A[mA]*exp(-s*hA[mA])
        
        Args:
            s (complex, float, int): s from complex plane
        
        Returns:
            dM (array): derivative of characteristic matrix M evaluated at s
        """
        return self.E + np.sum(self.A * self.hA * np.exp(-s*self.hA), axis=2)

    def print(self) -> None:
        """ Prints DDAE in readable form """

        with np.printoptions(precision=4, linewidth=1000, suppress=True):
            print(f"E 2x2 matrix")
            print(self.E)
            print("-"*50)
            for i in range(self.mA):
                print(f"A[:,:,{i} - tau={self.hA[i]}")
                print(self.A[:,:,i])
                print("-"*50)
            for i in range(self.mB):
                print(f"B[:,:,{i} - tau={self.hB[i]}")
                print(self.B[:,:,i])
                print("-"*50)
            for i in range(self.mC):
                print(f"C[:,:,{i} - tau={self.hC[i]}")
                print(self.C[:,:,i])
                print("-"*50)
            for i in range(self.mD):
                print(f"D[:,:,{i} - tau={self.hD[i]}")
                print(self.D[:,:,i])
                print("-"*50)