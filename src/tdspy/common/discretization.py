"""
TODO
"""

import logging
from typing import Type

import numpy as np
import scipy.special

from tdspy.base import TDSBase
from tdspy.dae import DAE


logger = logging.getLogger(__name__)


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


    if tds.mA == 1 and tds.hA[0] == 0.:
        # already DAE
        # TODO input output matrices
        return # TODO
    elif np.all(tds.hA == 0.):
        A = sum(tds.A)
        # TODO input output matrices
        return
    
    # unpack key tds matrices
    E = tds.E
    A = tds.A
    hA = tds.hA
    A_stacked = np.stack(A, axis=2)

    
    # main body
    if s0 != 0: # discretization around non-zero -> shift matrices
        logger.debug("")
        assert hA[0] == 0, "First delay assumed to be 0.0"
        A[0] -= s0*E
        for i in range(1,len(A)):
            A[i] *= np.exp(-s0 * hA[i])
    

    # continue line 86
    n_states = tds.n
    # n_inputs = tds.p2 # TODO
    hA_max = np.max(hA) # maximal delay - TODO assume sorted?

    if method == "cheb":
        logger.debug("Using Chebyshev polynomial of the first kind")
        # See original MATLAB docs [1, Eq. 2.11-2.13]
        #        [I*tau_m/2     0      -I*tau_m/4                                  ]
        #        [          I*tau_m/8      0      -I*tau_m/8                       ]
        # Pi_N = [   ....      ....       ....       ....       ....      ....     ]
        #        [                                           I*tau_m/(2*N)      0  ]     
        #        [    E         E          E          E         ....      ....   E ]
        diag_index = np.arange(0, N+1, 1, dtype=int)
        b = np.zeros(shape=(N,N+1))
        b[diag_index[:-1], diag_index[:-1]] = 1.0 / np.arange(1, N+1, 1) # main diagonal
        b[0,0] = 2
        b[diag_index[:-2], diag_index[2:]] = -1.0 / np.arange(1, N, 1) # offset diagonal
        #     [2  0  -1                         ]
        #     [0 0.5  0 -0.5                    ]
        # b = [0  0  1/3  0  -1/3               ]
        #     [                                 ]
        #     [                         -1/(N-1)]
        #     [                    1/N      0   ]
        b *= hA_max/4.0
        Pi_N = np.r_[np.kron(b, np.eye(n_states)),
                     np.kron(np.ones(shape=(1,N+1)), E)]
        
        #           [0  I 0  0 ... 0 ]
        #           [0  0 I  0 ... 0 ]
        # SIGMA_N = [  ...  ...  ... ]
        #           [0   ...   0   I ]
        #           [R0 R1    ...  RN]
        ##Sigma_N = np.c_[np.zeros(shape=(n_states*N, n_states)), np.eye(n_states*N)] TODO delete this row
        Sigma_N = np.diag(np.ones(N*n_states), k=n_states)
        #  Ri = A0 + sum_{k=1}^{mA} Ak Ti(-2*tau_k/tau_m+1) with Ti(.) the ith Chebyshev polynomial
        d = - np.transpose(hA) / hA_max * 2 + 1
        for i in range(N+1):
            Ri = np.sum(A_stacked * scipy.special.eval_chebyt(i, d), axis=2)
            Sigma_N[n_states*N:, i*n_states:(i+1)*n_states] = Ri

    elif method == "legendre":
        logger.debug("Using Legendre polynomial of the first kind")
        raise NotImplementedError("...")
        # scipy.special_eval_chebyt - first kind of chebyshev polynomial
        # scipy.special_eval_chebyu - second kind of chebyshev polynomial
        # scipy.special.eval_legendre
    
    # shift back if necessary
    if s0!=0:
        Sigma_N += s0 * Pi_N
    
    # Construct DAE
    dae = DAE(A=Sigma_N, B=None, C=None, D=None, E=Pi_N)
    return dae
