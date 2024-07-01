"""
TODO
"""

import logging
from typing import Type

import numpy as np
import numpy.typing as npt
import scipy.special

from tdspy.base import TDSBase
from tdspy.dae import DAE


logger = logging.getLogger(__name__)

def _discretize(E, A, hA, discretization: int, s0: complex=0j, method: str="cheb"):
    """ Discretizes DDAE into DAE (NO checks performed)

    DDAE of form:

        E x'(t) = A[0] x(t) + A[1] x(t-hA[1]) + .. + A[m] x(t-hA[m]),      (1)

    is discretized into DAE of form:

        E x'(t) = A x(t).                                                  (2)
    
    When method == 'cheb', the code uses the companion-type reformulation of
    the spectral discretisaion of the infinitesimal generator of the solution
    operator underlying the DDE as presented in Section 2.2 of the paper below.
    
    [1] Jarlebring, E., Meerbergen, K., & Michiels, W. (2010). A Krylov
        method for the delay eigenvalue problem. SIAM Journal on Scientific
        Computing, 32(6), pp. 3278-3300.  

    When method == 'legendre', the code uses a similar companion-type
    reformulation but now based on Legendre polynomials instead of Chebyshev
    polynomials.
    
    Args:
        E (array): right hand side matric of DDAE, assumed non-empty
        A (array): left hand side matrices of DDAE, assumed non-empty
        hA (array): vector of delays, assumed non-empty, hA[0] == 0
        discretization (int): degree of discretization > 0
        s0 (complex): point discretization is done around, default 0
        method (str): type of approximation, default 'cheb', allowed 'cheb',
            'legendre'

    Returns:
        tuple containing:

            - E (array): left hand-side matrix of DAE
            - A (array): right hand-side matrix of DAE
    """
    if s0 != 0: # discretization around non-zero -> shift matrices
        A[:,:,0] = A[:,:,0] - s0*E 
        A[:,:,1:] = A[:,:,1:] * np.exp(-s0 * hA[1:])
        for i in range(1,len(A)):
            A[i] = A[i] * np.exp(-s0 * hA[i])
    
    n_states = E.shape[0]
    hA_max = np.max(hA) # maximal delay - TODO assume sorted?

    if method == "cheb":
        logger.debug("Using Chebyshev polynomial of the first kind")
        # See original MATLAB docs [1, Eq. 2.11-2.13]
        #        [I*tau_m/2     0      -I*tau_m/4                                  ]
        #        [          I*tau_m/8      0      -I*tau_m/8                       ]
        # Pi_N = [   ....      ....       ....       ....       ....      ....     ]
        #        [                                           I*tau_m/(2*N)      0  ]     
        #        [    E         E          E          E         ....      ....   E ]
        diag_index = np.arange(0, discretization+1, 1, dtype=int)
        b = np.zeros(shape=(discretization, discretization+1))
        b[diag_index[:-1], diag_index[:-1]] = 1.0 / np.arange(1, discretization+1, 1) # main diagonal
        b[0,0] = 2
        b[diag_index[:-2], diag_index[2:]] = -1.0 / np.arange(1, discretization, 1) # offset diagonal
        #     [2  0    -1                     ]
        #     [0  1/2  0    -1/2              ]
        # b = [0  0    1/3  0    -1/3         ]
        #     [                               ]
        #     [                        1/(N-1)]
        #     [                  1/N      0   ]
        b *= hA_max/4.0
        Pi_N = np.r_[np.kron(b, np.eye(n_states)),
                     np.kron(np.ones(shape=(1,discretization+1)), E)]
        
        #           [0  I 0  0 ... 0 ]
        #           [0  0 I  0 ... 0 ]
        # SIGMA_N = [  ...  ...  ... ]
        #           [0   ...   0   I ]
        #           [R0 R1    ...  RN]
        Sigma_N = np.diag(np.ones(discretization*n_states), k=n_states).astype(np.complex128)
        #  Ri = A0 + sum_{k=1}^{mA} Ak Ti(-2*tau_k/tau_m+1) with Ti(.) the ith Chebyshev polynomial
        d = - np.transpose(hA) / hA_max * 2 + 1
        for i in range(discretization+1):
            Ri = np.sum(A * scipy.special.eval_chebyt(i, d), axis=2)
            Sigma_N[n_states*discretization:, i*n_states:(i+1)*n_states] = Ri

    elif method == "legendre":
        logger.debug("Using Legendre polynomial of the first kind")
        diag_index = np.arange(0, discretization+1, 1, dtype=int)
        b = np.zeros(shape=(discretization, discretization+1))
        b[diag_index[:-1], diag_index[:-1]] = 1.0 / (2*np.arange(0, discretization, 1)+1) # main diagonal
        b[diag_index[:-2], diag_index[2:]] = -1.0 / (2*np.arange(2, discretization+1, 1)+1) # offset diagonal
        #     [1  0    -1/5                           ]
        #     [0  1/3  0    -0.7                      ]
        # b = [0  0    1/5  0    -1/9                 ]
        #     [                                       ]
        #     [                             -1/(2*N+1)]
        #     [                    1/(2*N-1)    0     ]
        b *= hA_max / 2.
        Pi_N = np.r_[np.kron(b, np.eye(n_states)),
                     np.kron(np.ones(shape=(1,discretization+1)), E)]
        
        #           [0  I 0  0 ... 0 ]
        #           [0  0 I  0 ... 0 ]
        # SIGMA_N = [  ...  ...  ... ]
        #           [0   ...   0   I ]
        #           [R0 R1    ...  RN]
        Sigma_N = np.diag(np.ones(discretization*n_states), k=n_states).astype(np.complex128)
        #  Ri = A0 + sum_{k=1}^{mA} Ak Li(-2*tau_k/tau_m+1) with Li(.) the ith Legendre polynomial
        d = - np.transpose(hA) / hA_max * 2 + 1
        for i in range(discretization+1):
            Ri = np.sum(A * scipy.special.eval_legendre(i, d), axis=2)
            Sigma_N[n_states*discretization:, i*n_states:(i+1)*n_states] = Ri
    
    # shift back if necessary
    if s0 != 0:
        Sigma_N += s0 * Pi_N
    
    return Pi_N, Sigma_N # E, A

def discretize_ddae(E: npt.NDArray, A: npt.NDArray, hA: npt.NDArray, discretization: int, s0: complex=0j, method: str="cheb") -> tuple[npt.NDArray, npt.NDArray]:
    """ Discretizes DDAE into DAE

    DDAE of form:

        E x'(t) = A[0] x(t) + A[1] x(t-hA[1]) + .. + A[m] x(t-hA[m]),      (1)

    is discretized into DAE of form:

        E x'(t) = A x(t).                                                  (2)
    
    When method == 'cheb', the code uses the companion-type reformulation of
    the spectral discretisaion of the infinitesimal generator of the solution
    operator underlying the DDE as presented in Section 2.2 of the paper below.
    
    [1] Jarlebring, E., Meerbergen, K., & Michiels, W. (2010). A Krylov
        method for the delay eigenvalue problem. SIAM Journal on Scientific
        Computing, 32(6), pp. 3278-3300.  

    When method == 'legendre', the code uses a similar companion-type
    reformulation but now based on Legendre polynomials instead of Chebyshev
    polynomials.
    
    Args:
        E (array): right hand side matric of DDAE, assumed non-empty
        A (array): left hand side matrices of DDAE, assumed non-empty
        hA (array): vector of delays, assumed non-empty, hA[0] == 0
        discretization (int): degree of discretization > 0
        s0 (complex): point discretization is done around, default 0
        method (str): type of approximation, default 'cheb', allowed 'cheb',
            'legendre'

    Returns:
        tuple containing:

            - E (array): left hand-side matrix of DAE
            - A (array): right hand-side matrix of DAE
    """
    # TODO perform checks
    
    return _discretize(E, A, hA, discretization, s0, method)

def discretize(tds: Type[TDSBase], N: int, s0: complex=0j, method: str="cheb") -> DAE:
    """ Discretizes RDDE, NDDE or DDAE into DAE

    Discretizes Time-Delay System into Differential Algebraic Equation.

    Args:
        tds (RDDE, NDDE, DDAE): Definition of Time-Delay System
        N (int): degree of discretization N>0
        s0 (complex): point discretization is done around, default 0
        method (str): type of approximation, default 'cheb', allowed 'cheb', 'legendre'

    Returns:
        TODO
    """
    
    assert isinstance(N, int) and N > 6, "N has to be int > 6"
    assert isinstance(s0, (int, float, complex)), "s0 has to be float, int or complex"
    assert method in ["cheb", "legendre"], "allowed methods from ['cheb', 'legendre']"
    assert tds.hA[0] == 0, "First delay assumed to be 0.0" # TODO


    if tds.mA == 1 and tds.hA[0] == 0.:
        # already DAE
        # TODO input output matrices
        return # TODO
    elif np.all(tds.hA == 0.):
        A = sum(tds.A)
        # TODO input output matrices
        return
        
    Pi_N, Sigma_N = _discretize(tds.E, tds.A, tds.hA, N, s0, method)
    
    # Construct DAE
    dae = DAE(A=Sigma_N, B=None, C=None, D=None, E=Pi_N) # TODO update
    return dae
