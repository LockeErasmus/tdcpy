"""
Abstract Parent Class for RDDE, NDDE, DDAE
------------------------------------------
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

class TDSBase(ABC):

    tol_singular: float = 1e-12
    """ default tolerance for considering a matrix singular """

    @property
    @abstractmethod
    def mA(self) -> int:
        ...
    
    @property
    @abstractmethod
    def hA(self) -> npt.NDArray:
        ...

    @property
    @abstractmethod
    def A(self) -> npt.NDArray:
        ...

    @property
    @abstractmethod
    def E(self) -> npt.NDArray:
        ...

    @property
    @abstractmethod
    def n(self) -> int:
        """ state size """
        ...
    
    @abstractmethod
    def get_delay_difference_equation(self):
        """ associated delay difference equation """
        ...

    def _prepare_system_descriptor_matrix_vector(self, matrices: npt.NDArray | list[npt.NDArray],
                                                 delays: list[float]|npt.NDArray, allow_empty=True, allow_negative_delays=False, allow_complex=False, add_zero_delay=False, sort_by_delays=True, dtype=np.float64) -> tuple[npt.NDArray, npt.NDArray]:
        """ Prepare system descriptor matrices and delay vectors

        Args:
            matrices (npt.NDArray | list[npt.NDArray]): system descriptor matrices
            delays (list[float]|npt.NDArray): delay values

        """
        if delays is None:
            delays = np.zeros(shape=(0,), dtype=np.float64)

        if isinstance(delays, list):
            # note empty list of delays is allowed and converted properly to empty np array
            delays = np.array(delays, dtype=dtype)

        if matrices is None or (isinstance(matrices, list) and len(matrices) == 0): # None or empty list
            matrices = np.empty((0, 0, len(delays)), dtype=np.dtype)
        elif isinstance(matrices, list): # non-empty list
            # note: this correctly results in error if matrices have inconsistent shapes
            matrices = np.stack(matrices, axis=2) # results is 3D array

        if matrices.ndim != 3:
            raise ValueError("matrices must be a list of 2D arrays or a 3D array")
        
        if delays.ndim != 1:
            raise ValueError("delays must be a 1D array or a list of floats")
        
        if matrices.shape[2] != delays.shape[0]:
            raise ValueError("number of matrices must match number of delays")
        
        if not allow_empty and matrices.shape[2] == 0:
            raise ValueError("empty matrices/delays not allowed")
        
        if not allow_negative_delays and np.any(delays < 0):
            raise ValueError("negative delays not allowed")
        
        if add_zero_delay:
            if not np.any(delays == 0):
                # add zero delay and zero matrix to the beginning
                delays = np.r_[0.0, delays]
                zero_matrix = np.zeros((matrices.shape[0], matrices.shape[1], 1), dtype=matrices.dtype)
                matrices = np.concatenate((zero_matrix, matrices), axis=2)
        
        if sort_by_delays:
            sort_indices = np.argsort(delays)
            delays = delays[sort_indices]
            matrices = matrices[:, :, sort_indices]
        
        return matrices.astype(dtype=dtype), delays.astype(dtype=dtype)