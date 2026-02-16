# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

"""
NDDE - Neutral Delay Differential Equation
------------------------------------------
TODO TODO TODO
Notes:
    1. As of now, only A and hA, rest of the implementation will follow
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .common.compress import compress_matrices_delays, sort_matrices_delays
from .base import TDSBase
from .ddae import DDAE

logger = logging.getLogger(__name__)

class NDDE(TDSBase):
    """ Neutral Delay Differential Equaton

    dxdt(t) = A[i]*x(t - hA[i]) + ... + A[mA]*x(t - hA[mA])
              - H[0] * dxdt(t - hH[0]) - ... - H[mH] * dxdt(t - hH[mH])
    
    TODO documentation

    """
    def __init__(self, A: npt.NDArray, hA: npt.NDArray, H: npt.NDArray, hH: npt.NDArray,
                 B: npt.NDArray=None, hB: npt.NDArray=None, C: npt.NDArray=None, hC: npt.NDArray=None,
                 D: npt.NDArray=None, hD: npt.NDArray=None, **kwargs) -> None:
        """
        
        Args:
        
            **kwargs:
                dtype (): default np.float64
                tol_singular (float): considering value singular, default 1e-12

        
        """
        A, hA = self._prepare_system_descriptor_matrix_vector(A, hA, allow_empty=False,
                                                              allow_negative_delays=False,
                                                              allow_complex=False,
                                                              add_zero_delay=True,
                                                              sort_by_delays=True,
                                                              dtype=kwargs.get("dtype", np.float64))
        H, hH = self._prepare_system_descriptor_matrix_vector(H, hH, allow_empty=False,
                                                              allow_negative_delays=False,
                                                              allow_complex=False,
                                                              add_zero_delay=False,
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
        
        # --- ARGS ---
        ## dynamics
        self._A = A
        self._hA = hA
        self._H = H
        self._hH = hH
        self._B = B
        self._hB = hB
        self._C = C
        self._hC = hC
        self._D = D
        self._hD = hD

        # --- KWARGS ---
        self.dtype = kwargs.get("dtype", np.float64)
        self.tol_singular = kwargs.get("tol_singular", 1e-12)

        # TODO perform checks        
        # TODO rest of the system description

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
        """ NDDE can not be delay difference equation """
        return False
    
    def to_asymptotic_transfer_function(self, **kwargs) -> 'RDDE':
        raise NotImplementedError("Not implemented yet")
    
    def sort(self, inplace=False) -> 'NDDE':
        """ Sorts delays (self.hA) into ascending order """

        H, hH = None, None
        if self.H is not None and self.hH is not None:
            H, hH = sort_matrices_delays(self.H, self.hH)

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
            self._H = H
            self._hH = hH
            self._A = A
            self._hA = hA
            self._B = B
            self._hB = hB
            self._C = C
            self._hC = hC
            self._D = D
            self._hD = hD
        else:
            return NDDE(H=H, hH=hH, A=A, hA=hA, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)
    
    def compress(self, inplace=False) -> 'NDDE':
        """ Removes delay duplicates, sorts delays into ascending order
        
        Args:
            inplace (bool): wheter to modify existing NDDE or create a new one,
                default False
            rtol (float): relative tolerance for determining matrix element is
                zero, default 1e-5
            atol (float): absolute tolerance for determining matrix element is 
                zero, default 1e-8
        
        Returns:
            NDDE: compressed representation
            None: if inplace=True (current object is updated)
        """
        H, hH = None, None
        if self.H is not None and self.hH is not None:
            H, hH = compress_matrices_delays(self.H, self.hH)

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
            self._H = H
            self._hH = _H
            self._A = A
            self._hA = hA
            self._B = B
            self._hB = hB
            self._C = C
            self._hC = hC
            self._D = D
            self._hD = hD
        else:
            return NDDE(H=H, hH=hH, A=A, hA=hA, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)
    
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

    def eval_char_matrix(self, s:complex) -> npt.NDArray:
        """ Evaluate the characteristic matrix at `s` 
        
        M(s) = s*(E + H[0]*exp(-s*hH[0]) + ... + H[mH]*exp(-s*hH[mH])) - A[0]*exp(-s*hA[0]) - ... - A[mA]*exp(-s*hA[mA])
        
        Args: 
            s (complex, float, int): s from complex plane

        Returns:
            M (array): characteristic matrix M evaluated at s
        """
        return s*(self.E + np.sum(self.H * np.exp(-s*self.hH), axis=2)) - np.sum(self.A * np.exp(-s*self.hA), axis=2)
    
    def eval_char_matrix_derivative(self, s: complex) -> npt.NDArray:
        """ Derivative of the characteristic matrix with respect to s evaluated at s
        
        dM(s) = E + (H[0]*exp(-s*H[0])+...+H[mH]*exp(-s*H[mH]) - s*(hH[0]*H[0]*exp(-s*H[0])+...+hH[mH]*H[mH]*exp(-s*H[mH])
                    + (hA[0]*A[0]*exp(-s*hA[0])+...+hA[mA]*A[mA]*exp(-s*hA[mA]))

        Args: 
            s (complex, float, int: s from complex plane

        Returns: 
            dM(array): derivative of characteristic matrix M evaluated at s
        """
        return self.E + np.sum(self.H * np.exp(-s*self.hH), axis=2) - s*np.sum(self.H * self.hH * np.exp(-s*self.hH), axis=2)+ np.sum(self.A * self.hA * np.exp(-s*self.hA), axis=2)
    