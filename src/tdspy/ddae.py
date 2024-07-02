"""
DDAE implementation
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg


logger = logging.getLogger(__name__)


class DDAE:

    def __init__(self, A: npt.NDArray, hA: npt.NDArray, E=None, uE=None, vE=None,**kwargs) -> None:
        
        # assert A.size > 0, "A has to be non-empty array"        
        # assert np.atleast_3d(A) # TODO check
        # assert np.atleast_1d(hA) # TODO check
        assert np.all(hA >= 0.0), "Only positive delays possible"
        assert A.shape[2] == hA.shape[0], "number of delays does not match number of matrices Ai"
        # shape = A[0].shape

        # TODO if necessary, add 0 delay term
        # REALY ADD 0.0 delay term

        
        # ---
        self._A = A
        self._hA = hA

        self._E = E
        self._uE = uE
        self._vE = vE

        # --- KWARGS ---
        self.dtype = kwargs.get("dtype", np.float64)
        self.tol_singular = kwargs.get("tol_singular", 1e-12)

    @property
    def n(self) -> int:
        return self.A.shape[1]

    @property
    def E(self) -> npt.NDArray:
        if self._E is None:
            return np.eye(self.n, dtype=self.dtype)
        else:
            return self._E
    
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
        return self.hA.shape[0]

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
        """ Checks if DDAE uses complex storage for any of defining matrices """
        raise NotImplementedError()
    
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

            0 = A[0] x(t)
        """
        diff = self.get_delay_difference_equation()
        if diff.A.shape[2] == 0 and diff.hA[0] == 0:
            return True
        return False
        

    @property
    def is_essentially_neutral(self):
        """ Checks if DDAE is essentialy netural """
        return not self.is_essentially_retarded
    
    def _get_delay_difference_equation(self, uE: npt.NDArray, vE:npt.NDArray,
                                       normalize=False, tol=1e-14) -> tuple:
        """ Converts to Delay difference Equation Representation

        For a DDAE, the associated delay difference equation is given by
            U'*A[0]*V x(t-hA[0]) + ... + U'*A[mA-1]*V x(t-hA[mA-1]) = 0
        with U and V orthogonal matrices whose columns form a basis for the null
        space of E.

        Args:
            uE (array): assumed to be non-zero
            vE (array): assumed to be non-zero
            normalize (bool): whether to normalize delay difference equation,
                i.e. multiply by inv(A[0]), note that D0 and 0 delay term are
                omitted, optional, default False
            tol (float): norm tolerance for considering matrix vanish,'
                default 1e-14
        
        Returns:
            tuple containing

                - D (int): number of discretization points necessary
                - hD (complex): origin TODO
        """
        # calculate .. for both nullspaces, select the bigger one
        norm_uE = linalg.norm(uE, ord=1, axis=None)
        norm_vE = linalg.norm(vE, ord=1, axis=None)
        norm_null = max(norm_uE, norm_vE)

        # calculate Di = uE.T @ Ai @ vE, D.shape == A.shape (see numpy broadcasting)
        # D = np.transpose(np.transpose(np.transpose(uE) @ self.A) @ vE)
        D = []
        for i in range(self.A.shape[2]):
            D.append(uE.T @ self.A[:,:,i] @ vE)
        D = np.stack(D, axis=2)
        
        # select only Di =/= 0.0, i.e. Di sufficiently close to 0 are neglected
        mask = (linalg.norm(D, ord=1, axis=(0,1)) / norm_null) > tol
        D = D[:,:,mask]
        hD = self.hA[mask]

        # TODO, what if empty or 1 delay

        # normalize
        if normalize:
            hDD = hD[1:] # TODO assume at least 2 delays, i.e. [0, tau1]
            n, m = D.shape[0], hDD.shape[0]
            DD = np.zeros(shape=(n, n, m))
            
            if m == 1:
                DD[:,:,0] = linalg.lstsq(D[:,:,0], D[:,:,1])
            else:
                P, L, U = linalg.lu(D[:,:,0]) # LU decomposition for having inverse of A0
                for i in range(1, m+1):
                    DD[:,:, i-1] = linalg.lstsq(U, linalg.lstsq(L, P @ D[:,:,i])) # Di = inv(A0) @ Ai        
            
            return DD, hDD # return normalized
        else:
            return D, hD # return not normalized
   
    def get_delay_difference_equation(self, **kwargs) -> 'DDAE':
        """ Converts to Delay-difference Equation 

        For a DDAE, the associated delay difference equation is given by
            U'*A[0]*V x(t-hA[0]) + ... + U'*A[mA-1]*V x(t-hA[mA-1]) = 0
        with U and V orthogonal matrices whose columns form a basis for the null
        space of E.
        
        kwargs:
            tol (float): norm tolerance for considering matrix vanish,'
                default 1e-14
        
        Returns:
            DDAE representing delay difference equation if E is singular
            None if E is non-singular (there is no delay difference equation)

        """
        if self.is_logical:
            raise ValueError(f"Can't form DDE from logical")
                
        uE = self.uE # dynamic property -> calc it once and store into mem
        vE = self.vE # dynamic property -> calc it once and store into mem

        if np.size(uE) == 0:
            # TODO case where E is non-singular, return EMPTY DDAE
            return None
            # return DDAE(A=np.empty(shape=(0,0,0)), hA=np.empty(shape=(0,)))

        D, hD = self._get_delay_difference_equation(
            uE,
            vE,
            normalize=kwargs.get("normalize", False),
            tol= kwargs.get("tol", 1e-14),
        )
        nE = self.n
        dtype = self.E.dtype
        diff = DDAE(A=D, hA=hD, E=np.zeros(shape=(nE,nE), dtype=dtype),
                    uE=np.eye(nE, dtype=dtype), vE=np.eye(nE, dtype=dtype))
        return diff

    def to_asymptotic_transfer_function(self, **kwargs) -> 'DDAE':
        raise NotImplementedError("Not implemented yet")
    
    def sort(self, inplace=False):
        """ Sorts delays (mainly hA) into ascending order """
        delay_index = np.argsort(self.hA)

        # TODO also solve input matrices, output matrices

        if inplace:
            self._A = self.A[:,:, delay_index]
            self._hA = self.hA[delay_index]
        else:
            return DDAE(A=self.A[:,:, delay_index], hA=self.hA[delay_index])

    def compress(self, inplace=False, rtol=1e-5, atol=1e-8):
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
            return DDAE(E=self.E, A=A, hA=hA)
        
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

def normalize_delay_difference_equation(diff: DDAE, checkE=True):
    """ normalizes delay difference equation

    Transforms the delay difference equation such that the leading zero delay
    matrix A0 equals identity.

    Args:
        diff (DDAE): DDAE in form of delay difference equation (E=0)

    Returns:
        tuple containing

        - D (array): 3D array representing matrices [inv(A0)*A1, ... , inv(A0)*Am]
        - hDD (array): array of non-zero delays
    """
    print(diff.hA)
    hDD = diff.hA[1:] # TODO assume at least 2 delays, i.e. [0, tau1]
    print(hDD)

    DD = np.zeros(shape=(diff.n, diff.n, diff.mA-1))

    if diff.mA == 2:
        DD[:,:,0] = linalg.lstsq(diff.A[:,:,0], diff.A[:,:,1])
        return DD, hDD
    
    P, L, U = linalg.lu(diff.A[:,:,0]) # LU decomposition for having inverse of A0
    for i in range(1, diff.mA):
        DD[:,:, i-1] = linalg.lstsq(U, linalg.lstsq(L, P @ diff.A[:,:,i])) # Di = inv(A0) @ Ai
    return DD, hDD



