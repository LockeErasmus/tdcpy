
import numpy.typing as npt
from scipy import linalg


class DDAE:
    """ Delay Differential Algebraic Equation """

    
    def __init__(self, E: npt.NDArray):
        
        # TODO perform checks on inputs
        
        
        # TODO if uE, vE not defined, calculate them
        self._uE = linalg.null_space(E.T, rcond=self.tol_singular)
        self._vE = linalg.null_space(E, rcond=self.tol_singular)
        pass

    @property
    def tol_singular(self) -> float:
        """ relative tolerance for singular values (lower considered as 0.0) """
        return 1e-12
    
    @property
    def uE(self):
        """ orthonormal basis for left null space of E """
        return self._uE
    
    @property
    def vE(self):
        """ orthonormal basis for right null space of E """
        return self._vE
    