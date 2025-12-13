"""
DDAE implementation

.. todo::
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
    """ Delay Differential-Algebraic Equation (DDAE)

    .. math::

        E \dot{x}(t) = \sum_{k=1}^{m_A} A_k x(t - h_{A,k}) 
            + \sum_{k=1}^{m_B} B_k u(t - h_{B,k})
    
    .. math::
        
        y(t) = \sum_{k=1}^{m_C} C_k x(t - h_{C,k}) +
            \sum_{k=1}^{m_D} D_k u(t - h_{D,k})

    where :math:`x(t) \in \mathbb{R}^n` is the state vector, 
    :math:`u(t) \in \mathbb{R}^p` the input vector, and 
    :math:`y(t) \in \mathbb{R}^q` the output vector.

    
    """

    def __init__(self, A: npt.NDArray, hA: npt.NDArray, E: npt.NDArray=None,
                 B: npt.NDArray=None, hB: npt.NDArray=None, C: npt.NDArray=None, hC: npt.NDArray=None, 
                 D: npt.NDArray=None, hD: npt.NDArray=None, **kwargs) -> None:
        A, hA = self._prepare_system_descriptor_matrix_vector(A, hA, allow_empty=True,
                                                              allow_negative_delays=False,
                                                              allow_complex=False,
                                                              add_zero_delay=True,
                                                              sort_by_delays=True,
                                                              dtype=kwargs.get("dtype", np.float64))
        
        if E is not None:
            if not isinstance(E, np.ndarray):
                raise TypeError("E has to be ndarray")
            if E.ndim != 2:
                raise ValueError("E has to be 2D array")
            if E.shape[0] != A.shape[0] or E.shape[1] != A.shape[1]:
                raise ValueError("E shape does not match A shape")
        
        # for some legacy reasons, we allow uE, vE to be set directly
        # but we need to check their validity
        uE = kwargs.get("uE", None)
        vE = kwargs.get("vE", None)
        if uE is not None:
            if not isinstance(uE, np.ndarray):
                raise TypeError("uE has to be ndarray")
            if uE.ndim != 2:
                raise ValueError("uE has to be 2D array")
            if uE.shape[0] != A.shape[0]:
                raise ValueError("uE shape does not match A shape")
        if vE is not None:
            if not isinstance(vE, np.ndarray):
                raise TypeError("vE has to be ndarray")
            if vE.ndim != 2:
                raise ValueError("vE has to be 2D array")
            if vE.shape[0] != A.shape[1]:
                raise ValueError("vE shape does not match A shape")
            
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

        # solve other kwargs
        self.dtype = kwargs.get("dtype", np.float64)

        if kwargs.get("tol_singular", None) is not None:
            # set new tolerance for considering matrix singular
            if not isinstance(kwargs["tol_singular"], float):
                raise TypeError("tol_singular has to be float")
            if kwargs["tol_singular"] <= 0:
                raise ValueError("tol_singular has to be positive")
            
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
        """Check if DDAE is a normalized delay difference equation.
        
        A DDAE is normalized if it satisfies all of the following conditions:
        
        - DDAE is not logical
        - E == 0
        - A[0] == I
        
        Tolerances used for comparisons are absolute tolerance of 1e-12.
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
        """ Creates Delay-Difference Equation from DDAE
        
        For a DDAE, the associated delay difference equation is given by:
        add equation
        where U and V are orthogonal matrices whose columns form a basis for the null
        add equation
        
        Parameters
        ----------
        **kwargs
            Additional keyword arguments.
            
            tol : float, optional
                Norm tolerance for considering matrix vanish. Default is 1e-14.
            rcond : float, optional
                Relative condition number. Singular values s smaller than 
                rcond * max(s) are considered zero in null space construction. 
                Default is 1e-12.
        
        Returns
        -------
        DDAE or None
            DDAE representing delay difference equation or
            None if E is non-singular (there is no delay difference equation).
        
        Raises
        ------
        ValueError
            If the DDAE is logical. (not supported as of now, but is ready for future)
        """
        if self.is_logical:
            raise ValueError(f"Can't form delay difference equation from logical DDAE")
        
        D, hD = ddae_to_diff(self.E, self.A, self.hA, **kwargs)

        if np.size(D) == 0:
            # E is singular or close to being singular, this means that
            # DDAE is most likely essentialy retarded and we should return
            # empty delay difference equation. TODO: None is not great here
            return None
        
        nE = D.shape[1] # TODO - what if D.shape[1] == 0 ? can it happen?
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
