"""
Differential Algebraic Equation
"""

import logging

import numpy as np
import numpy.typing as npt


class DAE:
    """
    
    E x'(t) = A x(t) + B u(t)
       y(t) = C x(t) + D u(t) 
    
    """
    
    def __init__(self, A, B, C, D, E) -> None:
        # TODO checks
        self._A = A
        self._B = B
        self._C = C
        self._D = D
        self._E = E
    
    @property
    def n(self) -> int:
        if self.A.size == 0:
            return 0
        else:
            return self.A.shape[0]
    
    @property
    def A(self) -> npt.NDArray:
        return self._A
    
    @property
    def B(self) -> npt.NDArray:
        return self._B
    
    @property
    def C(self) -> npt.NDArray:
        return self._C
    
    @property
    def D(self) -> npt.NDArray:
        return self._D
    
    @property
    def E(self) -> npt.NDArray:
        return self._E