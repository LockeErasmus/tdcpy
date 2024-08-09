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
from .base import TDSBase
from .ddae import DDAE
from .common.composition import concatenate_2x2_by_delays

logger = logging.getLogger(__name__)


class ClosedLoop(TDSBase):
    """ Class representing system - controller interconnection

    """
    def __init__(self, system: DDAE, order: int, y_indices: list=None, u_indices: list=None, K0: npt.NDArray=None, hK: npt.NDArray = None, **kwargs) -> None:
        """ TODO


        Args:
            order (int): order of controller, non-negative
        """

        # TODO perform checks
        assert isinstance(order, int)
        assert order >= 0
        
        self._system = system
        self._order = order # order of controller
        if y_indices is None:
            y_indices = [0]
        if u_indices is None:
            u_indices = [0]
        self._y_indices = y_indices # system exposed measurements
        self._u_indices = u_indices # system outputs exposed to control

        if hK is None:
            hK = np.array([0.])
        
        self._hK = hK
        
        if K0 is None:
            # TODO run heuristic for initializing controller
            #K0 = np.random.rand() # TODO
            raise NotImplementedError(".")
        
        self._K = K0 # TODO

        # CL representation matrices -> TODO move to separate function
        dtype = np.float64

        ndelaysA = system.mA + system.mB + system.mC + system.mD
        w_indices = [i for i in range(system.B.shape[1]) if i not in u_indices]
        z_indices = [i for i in range(system.C.shape[0]) if i not in y_indices]
        len_u = len(u_indices)
        len_w = len(w_indices)
        len_y = len(y_indices)
        len_z = len(z_indices)
        n = system.A.shape[0] + system.B.shape[1] + system.C.shape[0] + order
        
        E = np.zeros(shape=(n, n), dtype=dtype)
        A = np.zeros(shape=(n, n, ndelaysA), dtype=dtype)
        hA = np.zeros(shape=(ndelaysA,), dtype=dtype)
        B = np.zeros(shape=(n, len_w, 1), dtype=dtype)
        hB = np.zeros(shape=(1,), dtype=dtype)
        C = np.zeros(shape=(len_z, n, 1), dtype=dtype)
        hC = np.zeros(shape=(1,), dtype=dtype)
        D = np.zeros(shape=(len_z, len_w, 1), dtype=dtype)
        hD = np.zeros(shape=(1,), dtype=dtype)

        # place system matrices
        rows_end = system.A.shape[0] + len_y + len_z
        cols_end = system.A.shape[1] + len_u + len_w
        reordered_output_indices = np.r_[self.y_indices, self.z_indices]
        reordered_input_indices = np.r_[self.u_indices, self.w_indices]

        concatenate_2x2_by_delays(
            self.system.E, self.system.A,
            self.system.B[np.ix_(range(system.B.shape[0]), reordered_input_indices, range(system.B.shape[2]))],
            self.system.C[np.ix_(reordered_output_indices, range(self.system.C.shape[1]), range(self.system.C.shape[2]))],
            self.system.D[np.ix_(reordered_output_indices, reordered_input_indices, range(self.system.D.shape[2]))],
            system.hA, system.hB, system.hC, system.hD,
            EE = E[:rows_end, :cols_end],
            AA = A[:rows_end, :cols_end, :],
            hAA= hA,
        )

        # interconnect system and controller
        # here I assume that hA[0], other approach would be to concatenate
        assert hA[0] == 0 # this should be always fullfilled
        n_auxiliary = len_u + len_w + len_z + len_y # number of auxiliary variables
        indices = np.array([i for i in range(self.system.A.shape[0], self.system.A.shape[0]+n_auxiliary)], dtype=int)
        A[indices, indices[::-1], 0] = -1

        # BB and CC matrix, such that A[:,:,i] = P[:,:,i] + BB @ K[:,:,i] @ CC
        BB = np.zeros(shape=(n, order+len_u), dtype=dtype)
        BB[-order-len_u:,:] = np.eye(order+len_u)[::-1,:]
        CC = np.zeros(shape=(order+len_y, n), dtype=dtype)
        CC[:,-order-len_y:] = np.eye(order+len_y)[::-1,:]

        # add controller LHS
        if order > 0:
            E[-order:, -order:,] = np.eye(order, dtype=dtype)

        # B,C,D matrices
        B[system.A.shape[0] + len_y + len_z:system.A.shape[0] + len_y + len_z + len_w, :, 0] = np.eye(len_w)
        C[:, system.A.shape[1] + len_u + len_w:system.A.shape[1] + len_u + len_w + len_z,0] = np.eye(len_z)
        # D matrix is just zeros

        A, hA = compress_matrices_delays(A, hA)

        # fill variables
        self._E = E
        self._A = A
        self._hA = hA
        self._B = B
        self._hB = hB
        self._C = C
        self._hC = hC
        self._D = D
        self._hD = hD

        # matrices BB CC to class namespace
        self._BB = BB
        self._CC = CC
        


    @property
    def n(self) -> int:
        """ number of variables """
        return self.system.A.shape[0] + self.system.B.shape[1] + self.system.C.shape[0] + self.controller_order
    
    @property
    def n_inputs(self) -> int:
        """ number of inputs """
        return self.B.shape[1]

    @property
    def n_outputs(self) -> int:
        """ number of outputs """
        return self.C.shape[0]
    
    @property
    def E(self) -> npt.NDArray:
        return self._E
    
    @property
    def A(self) -> npt.NDArray:
        """ dynamics matrices """
        A = np.concatenate(
            [
                self._A,
                np.stack([self.BB @ self.K[:,:,i] @ self.CC for i in range(self.K.shape[2])], axis=2)
            ],
            axis=2,
        )
        return A
    
    @property
    def hA(self) -> npt.NDArray:
        """ dynamic delays """
        return np.r_[self._hA, self.hK]
    
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
        """ number of output delays """
        return self.hD.shape[0]
    
    def get_delay_difference_equation(self):
        raise NotImplementedError(".")
    
    # CLOSED-LOOP SPECIFIC
    @property
    def system(self) -> DDAE:
        """ Original system we are controlling """
        # TODO maybe also RDDE NDDE
        return self._system
    
    @property
    def BB(self) -> npt.NDArray:
        return self._BB
    
    @property
    def CC(self) -> npt.NDArray:
        return self._CC

    @property
    def y_indices(self) -> list[int]:
        return self._y_indices
    
    @property
    def u_indices(self) -> list[int]:
        return self._u_indices
    
    @property
    def w_indices(self) -> list[int]:
        return [i for i in range(self.system.B.shape[1]) if i not in self.u_indices]
    
    @property
    def z_indices(self) -> list[int]:
        return [i for i in range(self.system.C.shape[0]) if i not in self.y_indices]

    # CONTROLER SPECIFIC
    @property
    def controller(self) -> DDAE:
        """ Returns Controller as DDAE """

        K = self.K
        hK = self.hK
        nc = self.controller_order
        
        ddae = DDAE(
            E=np.eye(self.controller_order),
            A=K[:nc,:nc,:],
            hA=hK,
            B=K[:nc,nc:,:],
            hB=hK,
            C=K[nc:,:nc,:],
            hC=hK,
            D=K[nc:,nc:,:],
            hD=hK,
        )
        return ddae

    @property
    def n_controller_inputs(self) -> int:
        return len(self._y_indices)
    
    @property
    def n_controller_outputs(self) -> int:
        return len(self._u_indices)
    
    @property
    def controller_order(self) -> int:
        return self._order

    @property
    def K(self):
        return self._K
    
    @property
    def hK(self):
        return self._hK
    
    @property
    def mK(self):
        return self._hK.shape[0]
    
    @K.setter
    def K(self, value: npt.NDArray):
        """ Set new controller values """
        expected_shape = (
            self.controller_order + self.n_controller_outputs,
            self.controller_order + self.n_controller_inputs,
            self.mK,
        )
        assert value.shape == expected_shape # TODO error not assert
        self._K = value

