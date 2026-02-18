# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

"""
Set of high level API functions for creating controllers


TODO:
    1. implementation of low level functions as well (matrices)
    2. as of now, DDAE later make it work for all three types of TDS
"""

import logging

import numpy as np
import numpy.typing as npt

from .rdde import RDDE
from .ndde import NDDE
from .ddae import DDAE
from .common.composition import concatenate_2x2_by_delays
from .common.compress import compress_matrices_delays
from .common.closed_loop import controller_reprezentation


logger = logging.getLogger(__name__)



def _handle_indices_defaults(u_indices: list[int] | None, y_indices: list[int] | None,
                             n_inputs: int, n_outputs: int) -> tuple[list[int], list[int]]:
    """ Checks and solves default indices in case of interconnection of two
    systems
    """
    print(f"{u_indices=}, {y_indices=}, {n_inputs=}, {n_outputs=}")    
    # 4 cases possible
    if y_indices and u_indices: # 1: both defined and possible -> done
        pass    
    elif y_indices and u_indices is None: # 2: defined outputs -> try to match to first n inputs
        if len(y_indices) > n_outputs:
            raise ValueError(f"Can not solve defaults for {y_indices} --> [?] not enough inputs")
        u_indices = [ix for ix in range(len(y_indices))]
    elif u_indices and y_indices is None: # 3: defined inputs -> try to match to first n outputs
        if len(u_indices) > n_inputs:
            raise ValueError(f"Can not solve defaults for [?] --> {u_indices} not enough outputs")
        y_indices = [ix for ix in range(len(u_indices))]
    else: # 4: both are None -> 2 cases
        u_indices = [ix for ix in range( min(n_inputs, n_outputs) )]
        y_indices = u_indices[:] # shallow copy is fine since only list of int

    # check if possible
    if any( ix >= n_inputs for ix in u_indices ) or any( ix >= n_outputs for ix in y_indices ):
        raise ValueError(f"Interconnection indices can not be >= than respective matrix shape")

    return u_indices, y_indices


def interconnect(tds1: DDAE, tds2: DDAE, y1_indices: list=None, u2_indices:list = None,
                 y2_indices: list=None, u1_indices:list = None, **kwargs) -> DDAE:
    """ Creates and interconnected system from two systems and interconnection indices

    Assume we have two systems:

    .. code-block:: text

        E1 dx1dt = SUM A1[i] x1(t-hA1[i]) + SUM B1[j] u1(t-hB1[j])
              y1 = SUM C1[k] x1(t-hC1[k]) + SUM D1[l] u1(t-hD1[l])

        E2 dx2dt = SUM A2[i] x2(t-hA2[i]) + SUM B2[j] u2(t-hB2[j])
              y2 = SUM C2[k] x2(t-hC2[k]) + SUM D2[l] u2(t-hD2[l])
        
        And interconnection defined via indices mapping, then the final system can
        be discribed via TODO


        x* = [x1, u1, y1, x2, u2, y2]

            E1, 0, 0,  0, 0, 0
            0, 0, 0,  0, 0, 0
        E = 0, 0, 0,  0, 0, 0
            0, 0, 0, E2, 0, 0
            0, 0, 0,  0, 0, 0
            0, 0, 0,  0, 0, 0
    
    TODO create scheme and maybe even equations

    Parameters
    ----------

    tds1 : DDAE
        system 1 to be interconnected
    tds2 : DDAE
        system 2 to be interconnected
    y1_indices : list, optional
        list of indices (outputs of system 1), if not defined, [0] is assumed
    u2_indices : list, optional
        list of indices (inputs of system 2), if not defined, [0] is assumed
    y2_indices : list, optional
        list of indices (outputs of system 2), if not defined, [0] is assumed
    u1_indices : list, optional
        list of indices (inputs of system 1), if not defined, [0] is assumed
    **kwargs:
        compress (bool): perform compression of resulting system, default True

        
    Returns
    -------
    interconnected system: DDAE

    Notes
    -----

    Examples
    --------
    >>> from tdcpy.controller import interconnect
    >>> from tdcpy.ddae import DDAE
    >>> A1 = np.array([[[1.0]]])
    >>> hA1 = np.array([0.0])
    >>> E1 = np.eye(1)
    >>> B1 = np.array([[[1.0]]])
    >>> hB1 = np.array([0.0])
    >>> C1 = np.array([[[1.0]]])
    >>> hC1 = np.array([0.0])
    >>> D1 = np.array([[[0.0]]])
    >>> hD1 = np.array([0.0])
    >>> tds1 = DDAE(A=A1, hA=hA1, E=E1, B=B1, hB=hB1, C=C1, hC=hC1, D=D1, hD=hD1)
    >>> A2 = np.array([[[0.5]]])
    >>> hA2 = np.array([0.0])
    >>> E2 = np.eye(1)
    >>> B2 = np.array([[[1.0]]])
    >>> hB2 = np.array([0.0])
    >>> C2 = np.array([[[1.0]]])
    >>> hC2 = np.array([0.0])
    >>> D2 = np.array([[[0.0]]])
    >>> hD2 = np.array([0.0])
    >>> tds2 = DDAE(A=A2, hA=hA2, E=E2, B=B2, hB=hB2, C=C2, hC=hC2, D=D2, hD=hD2)
    >>> interconnected = interconnect(tds1, tds2)
    >>> interconnected.E.shape
    (6, 6)
    """

    # perform checks

    # all IO matrices defined
    # correct and possible shapes

    # Extract matrices and delays
    E1, A1, B1, C1, D1 = tds1.E, tds1.A, tds1.B, tds1.C, tds1.D
    hA1, hB1, hC1, hD1 = tds1.hA, tds1.hB, tds1.hC, tds1.hD
    E2, A2, B2, C2, D2 = tds2.E, tds2.A, tds2.B, tds2.C, tds2.D
    hA2, hB2, hC2, hD2 = tds2.hA, tds2.hB, tds2.hC, tds2.hD
    
    # TODO argument names and make sure it is list
    # indices all are constructed as 1D arrays and axes are added later to leverage
    # broadcasting, it is necessary to have indices as dtype=int (alternatively as boolean mask)
    
    # First, handle defaults: assume tds2 is controller and if not defined, try to connect all
    u1_indices, y2_indices = _handle_indices_defaults(u1_indices, y2_indices, B1.shape[1], D2.shape[0])
    u2_indices, y1_indices = _handle_indices_defaults(u2_indices, y1_indices, B2.shape[1], D1.shape[0])
        
    ## System 1
    u1_indices = np.array(u1_indices, dtype=int)
    w1_indices = np.array([i for i in range(B1.shape[1]) if i not in u1_indices], dtype=int)
    y1_indices = np.array(y1_indices, dtype=int)
    z1_indices = np.array([i for i in range(D1.shape[0]) if i not in y1_indices], dtype=int)

    ## System 2
    u2_indices = np.array(u2_indices, dtype=int)
    w2_indices = np.array([i for i in range(B2.shape[1]) if i not in u2_indices], dtype=int)
    y2_indices = np.array(y2_indices, dtype=int)
    z2_indices = np.array([i for i in range(D2.shape[0]) if i not in y2_indices], dtype=int)

    # log interconection indices
    logger.debug(f"Mapping TDS1 outputs {y1_indices} to TDS2 inputs {u2_indices}")
    logger.debug(f"Mapping TDS2 outputs {y2_indices} to TDS1 inputs {u1_indices}")

    # interconnection algorithm
    # TODO: separate function on matrices
    # For now assume NO MATRICES are empty
    len_u1, len_w1 = len(u1_indices), len(w1_indices)
    len_y1, len_z1 = len(y1_indices), len(z1_indices)
    len_u2, len_w2 = len(u2_indices), len(w2_indices)
    len_y2, len_z2 = len(y2_indices), len(z2_indices)

    # TODO: for now, log table
    nrows = A1.shape[0] + len_u1 + len_y1 + A2.shape[0] + len_u2 + len_y2
    ncols = A1.shape[1] + len_u1 + len_y1 + A2.shape[1] + len_u2 + len_y2
    ndelays1 = hA1.shape[0] + hB1.shape[0] + hC1.shape[0] + hD1.shape[0]
    ndelays2 = hA2.shape[0] + hB2.shape[0] + hC2.shape[0] + hD2.shape[0]
    ndelays = ndelays1 + ndelays2 # TODO rename to ndelaysA
    ndelaysB = hB1.shape[0] + hD1.shape[0] + hB2.shape[0] + hD2.shape[0]
    ndelaysC = hC1.shape[0] + hD1.shape[0] + hC2.shape[0] + hD2.shape[0]
    ndelaysD = hD1.shape[0] + hD2.shape[0]

    # initialize E, A, hA, B, hB
    E = np.zeros(shape=(nrows, ncols), dtype=E1.dtype)
    A = np.zeros(shape=(nrows, ncols, ndelays), dtype=A1.dtype)
    hA = np.zeros(shape=(ndelays,), dtype=hA1.dtype)
    B = np.zeros(shape=(nrows, len_w1 + len_w2, ndelaysB), dtype=B1.dtype)
    hB = np.zeros(shape=(ndelaysB,), dtype=hB1.dtype)
    C = np.zeros(shape=(len_z1 + len_z2, ncols, ndelaysC), dtype=C1.dtype)
    hC = np.zeros(shape=(ndelaysC,), dtype=hC1.dtype)
    D = np.zeros(shape=(len_z1 + len_z2, len_w1 + len_w2, ndelaysD), dtype=D1.dtype)
    hD = np.zeros(shape=(ndelaysD,), dtype=hD1.dtype)

    # place first system
    rows_start1 = 0 # equations
    rows_end1 = rows_start1 + A1.shape[0] + len_y1
    cols_start1 =  0 # variables
    cols_end1 = cols_start1 + A1.shape[1] + len_u1
    concatenate_2x2_by_delays(
        E1, A1,
        B1[np.ix_(range(B1.shape[0]), u1_indices, range(B1.shape[2]))],
        C1[np.ix_(y1_indices, range(C1.shape[1]), range(C1.shape[2]))],
        D1[np.ix_(y1_indices, u1_indices, range(D1.shape[2]))],
        hA1, hB1, hC1, hD1,
        EE = E[rows_start1:rows_end1, cols_start1:cols_end1],
        AA = A[rows_start1:rows_end1, cols_start1:cols_end1, :ndelays1],
        hAA= hA[:ndelays1],
    )

    # place second system
    rows_start2 = A1.shape[0] + len_y1 + len_u2 # equations
    rows_end2 = rows_start2 + A2.shape[0] + len_y2
    cols_start2 =  A1.shape[1] + len_u1 + len_y1 # variables
    cols_end2 = cols_start2 + A2.shape[1] + len_u2
    concatenate_2x2_by_delays(
        E2, A2, 
        B2[np.ix_(range(B2.shape[0]), u2_indices, range(B2.shape[2]))],
        C2[np.ix_(y2_indices, range(C2.shape[1]), range(C2.shape[2]))],
        D2[np.ix_(y2_indices, u2_indices, range(D2.shape[2]))],
        hA2, hB2, hC2, hD2,
        EE = E[rows_start2:rows_end2,cols_start2:cols_end2],
        AA = A[rows_start2:rows_end2,cols_start2:cols_end2, ndelays1:],
        hAA= hA[ndelays1:],
    )

    # interconnect systems
    # here I assume that hA[0], other approach would be to concatenate
    assert hA[0] == 0
    A[A1.shape[0]:A1.shape[0]+len_y1, cols_end1:cols_end1+len_y1,0] = -np.eye(len_y1) # equation 0 = C1 @ x1 + D1 @ u1 - I @ y1
    A[rows_end1:rows_end1+len_y1, cols_end1:cols_end1+len_y1,0] = np.eye(len_y1) # equation 0 = I @ y1 - I @ u2 - PART y1
    A[rows_end1:rows_end1+len_y1, cols_start2+A2.shape[1]:cols_start2+A2.shape[1]+len_u2,0] = -np.eye(len_u2) # equation 0 = I @ y1 - I @ u2 - PART u2
    A[rows_start2+A2.shape[0]:rows_start2+A2.shape[0]+len_y2, cols_end2:cols_end2+len_y2,0] = -np.eye(len_y2) # equation 0 = C2 @ x2 + D2 @ u2 - I @ y2
    A[rows_end2:rows_end2+len_y2, cols_end2:cols_end2+len_y2,0] = np.eye(len_y2) # equation 0 = I @ y2 - I @ u1 - PART y2
    A[rows_end2:rows_end2+len_u1, A1.shape[1]:A1.shape[1]+len_u1,0] = -np.eye(len_u1) # equation 0 = I @ y2 - I @ u1 - PART u1

    # Populate B, hB
    n, m, p, q = hB1.shape[0], hD1.shape[0], hB2.shape[0], hD2.shape[0]
    B[:B1.shape[0], :len_w1, :n] = B1[np.ix_(range(B1.shape[0]), w1_indices, range(B1.shape[2]))] # B1: w1 -> x1
    B[B1.shape[0]: B1.shape[0]+len_y1, :len_w1, n:n+m] = D1[np.ix_(y1_indices, w1_indices, range(D1.shape[2]))] # D1: w1 -> y1
    B[rows_start2:rows_start2+B2.shape[0], len_w1:,n+m:n+m+p] = B2[np.ix_(range(B2.shape[0]), w2_indices, range(B2.shape[2]))] # B2: w2 -> x2
    B[rows_start2+B2.shape[0]:rows_start2+B2.shape[0]+len_y2, len_w1:,n+m+p:] = D2[np.ix_(y2_indices, w2_indices, range(D2.shape[2]))] # D2: w2 -> y2
    
    hB[:n] = hB1
    hB[n:n+m] = hD1
    hB[n+m:n+m+p] = hB2
    hB[n+m+p:] = hD2

    # Populate C, hC
    n, m, p, q = hC1.shape[0], hD1.shape[0], hC2.shape[0], hD2.shape[0]
    C[:len_z1, :C1.shape[1], :n] = C1[np.ix_(z1_indices, range(C1.shape[1]), range(C1.shape[2]))] # C1: x1 -> z1
    C[:len_z1, C1.shape[1]:C1.shape[1]+len_u1, n:n+m] = D1[np.ix_(z1_indices, u1_indices, range(D1.shape[2]))] # D1: u1 -> z1
    C[len_z1:, cols_start1:cols_start1+C2.shape[1], n+m:n+m+p] = C2[np.ix_(z2_indices, range(C2.shape[1]), range(C2.shape[2]))] # C2: x2 -> z2
    C[len_z1:, cols_start1+C2.shape[1]:cols_start1+C2.shape[1]+len_u2, n+m+p:] = D2[np.ix_(z2_indices, u2_indices, range(D2.shape[2]))] # D2: u2 -> z2
    
    hC[:n] = hC1
    hC[n:n+m] = hD1
    hC[n+m:n+m+p] = hC2
    hC[n+m+p:] = hD2

    # Populate D, hD
    # m, q reused from before: m=hD1.shape[0], q=hD2.shape[0]
    D[:len_z1, :len_w1, :m] = D1[np.ix_(z1_indices, w1_indices, range(D1.shape[2]))] # D1: w1 -> z1
    D[len_z1:, len_w1:, m:] = D2[np.ix_(z2_indices, w2_indices, range(D2.shape[2]))] # D2: w2 -> z2
    hD[:m] = hD1
    hD[m:] = hD2

    if kwargs.get("compress", True):
        A, hA = compress_matrices_delays(A, hA)
        B, hB = compress_matrices_delays(B, hB)
        C, hC = compress_matrices_delays(C, hC)
        D, hD = compress_matrices_delays(D, hD)

    return DDAE(A=A, hA=hA, E=E, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)

def create_static_controller(K: npt.NDArray) -> DDAE:
    """ Creates static controller from the matrix of coefficients 
    
    Static controller is assumed to be of a form
        y = K*u,
    but is constructed as DDAE with all delay equal to 0.0 and matrices A, B, C
    empty, i.e.

        I dxdt = A*x + B*u
             y = C*x + K*u
    
    Parameters
    ----------
    K : array
        1D or 2D array representing static controller gains

    Returns
    -------
    DDAE
        DDAE representation of static controller

    Notes
    -----
    
    Examples
    --------
    >>> from tdcpy.controller import create_static_controller
    >>> K = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> controller = create_static_controller(K)
    >>> controller.A.shape
    (0, 0, 1)
    >>> controller.B.shape
    (0, 2, 0)
    >>> controller.C.shape
    (2, 0, 0)
    >>> controller.D.shape
    (2, 2, 1)
    """

    assert isinstance(K, np.ndarray), "K is assumed to be array"
    assert K.size > 0, "K is assumed not to be empty"
    assert K.ndim == 1 or K.ndim == 2, "K has to be 2D array"
    
    if K.ndim == 1:
        logger.warning(f"Provided K is 1D vector, I will assume you wanted to create controller with n inputs (len of K) and 1 output.")
        K = K[np.newaxis, :]
    
    n, m = K.shape # 

    A, B = np.zeros(shape=(0, 0, 0)), np.zeros(shape=(0, m, 0))
    C, D = np.zeros(shape=(n, 0, 0)), np.stack([K], axis=2)

    hA, hB = np.zeros(shape=(0,)), np.zeros(shape=(0,))
    hC, hD = np.zeros(shape=(0,)), np.array([0.0])

    return DDAE(A=A, hA=hA, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)

def create_dynamic_controller(A: npt.NDArray, B: npt.NDArray, C: npt.NDArray,
                              D: npt.NDArray) -> DDAE:
    """ Creates dynamic controller from state space representation

    The form is assumed to be

        I dxdt = A*x + B*u
             y = C*x + K*u
    
    Parameters  
    ----------
    A : array
        state matrix, shape (n, n)
    B : array
        input matrix, shape (n, m)
    C : array
        output matrix, shape (p, n)
    D : array
        feedthrough matrix, shape (p, m)

    Returns
    -------
    ddae : DDAE
        DDAE representation of dynamic controller

    Notes
    -----

    Examples
    --------
    >>> from tdcpy.controller import create_dynamic_controller
    >>> A = np.array([[0.0, 1.0], [-2.0, -3.0]])
    >>> B = np.array([[0.0], [1.0]])
    >>> C = np.array([[1.0, 0.0]])
    >>> D = np.array([[0.0]])
    >>> controller = create_dynamic_controller(A, B, C, D)
    >>> controller.A.shape
    (2, 2, 1)
    >>> controller.B.shape
    (2, 1, 1)
    >>> controller.C.shape
    (1, 2, 1)
    >>> controller.D.shape
    (1, 1, 1)

    """

    assert isinstance(A, np.ndarray), "A is assumed to be array"
    assert A.size > 0, "A is assumed not to be empty"
    assert A.ndim == 2, "A has to be 2D array"
    assert isinstance(B, np.ndarray), "B is assumed to be array"
    assert B.size > 0, "B is assumed not to be empty"
    assert B.ndim == 2, "B has to be 2D array"
    assert isinstance(C, np.ndarray), "A is assumed to be array"
    assert C.size > 0, "C is assumed not to be empty"
    assert C.ndim == 2, "C has to be 2D array"
    assert isinstance(D, np.ndarray), "A is assumed to be array"
    assert D.size > 0, "D is assumed not to be empty"
    assert D.ndim == 2, "D has to be 2D array"

    # dimension check is performed in DDAE constructor
    # assert A.shape[0] == B.shape[0]
    # assert A.shape[1] == C.shape[1]
    # assert C.shape[0] == D.shape[0]
    # assert B.shape[1] == D.shape[1]

    controller = DDAE(
        A=A[:,:,np.newaxis], hA=np.array([0.0]),
        B=B[:,:,np.newaxis], hB=np.array([0.0]),
        C=C[:,:,np.newaxis], hC=np.array([0.0]),
        D=D[:,:,np.newaxis], hD=np.array([0.0]),
    )

    return controller

def interconnect2(tds1: DDAE, y1_indices: list=None, u1_indices:list = None, 
                  hA2: npt.NDArray = None, hB2: npt.NDArray = None, 
                  hC2: npt.NDArray = None, hD2: npt.NDArray = None,
                  **kwargs) -> DDAE:
    """ Creates and interconnected system ready for stabilitzation

    Parameters
    ----------
    tds1 : DDAE
         system 1 to be interconnected
    y1_indices : list, optional
        indices of measurements
    u1_indices : list, optional
        indices of controlled inputs
    hA2 : array, optional
        controller delays, default None will assume delay vector to be [0.0]
    hB2 : array, optional
        controller delays, default None will assume delay vector to be [0.0]
    hC2 : array, optional
        controller delays, default None will assume delay vector to be [0.0]
    hD2 : array, optional
        controller delays, default None will assume delay vector to be [0.0]

    Returns
    -------
    ddae : DDAE
        interconnected system 

    Examples
    --------
    >>> from tdcpy.controller import interconnect2
    >>> from tdcpy.ddae import DDAE
    >>> A1 = np.array([[[1.0]]])
    >>> hA1 = np.array([0.0])
    >>> E1 = np.eye(1)
    >>> B1 = np.array([[[1.0]]])
    >>> hB1 = np.array([0.0])
    >>> C1 = np.array([[[1.0]]])
    >>> hC1 = np.array([0.0])
    >>> D1 = np.array([[[0.0]]])
    >>> hD1 = np.array([0.0])
    >>> tds1 = DDAE(A=A1, hA=hA1, E=E1, B=B1, hB=hB1, C=C1, hC=hC1, D=D1, hD=hD1)
    >>> interconnected = interconnect2(tds1)
    >>> interconnected.E.shape
    (5, 5)

    """
    # perform checks

    # all IO matrices defined
    # correct and possible shapes

    # First, handle defaults
    if y1_indices is None:
        y1_indices = [0]
    if u1_indices is None:
        u1_indices = [0]

    # controller packed into TODO
    n_controller = 0 # order of controller, 0, 1, ..., inf
    n_inputs = len(y1_indices) # assumed to be at least 1
    n_outputs = len(u1_indices) # assumed to be at least 1
    E2, K, hK = controller_reprezentation(order=0, n_inputs=n_inputs, n_outputs=n_outputs)

    nk, mk = n_controller + n_inputs, n_controller + n_inputs

    # Extract matrices and delays of a system
    E1, A1, B1, C1, D1 = tds1.E, tds1.A, tds1.B, tds1.C, tds1.D
    hA1, hB1, hC1, hD1 = tds1.hA, tds1.hB, tds1.hC, tds1.hD    
    
    # TODO argument names and make sure it is list
    # indices all are constructed as 1D arrays and axes are added later to leverage
    # broadcasting, it is necessary to have indices as dtype=int (alternatively as boolean mask)
    
    ## System 1
    u1_indices = np.array(u1_indices, dtype=int)
    w1_indices = np.array([i for i in range(B1.shape[1]) if i not in u1_indices], dtype=int)
    y1_indices = np.array(y1_indices, dtype=int)
    z1_indices = np.array([i for i in range(D1.shape[0]) if i not in y1_indices], dtype=int)

    ## Controller
    u2_indices = np.array([i for i in range(n_inputs)], dtype=int)
    w2_indices = np.array([], dtype=int) # empty by definition
    y2_indices = np.array([i for i in range(n_outputs)], dtype=int)
    z2_indices = np.array([], dtype=int) # empty by definiiton

    # log interconection indices
    logger.debug(f"Mapping system outputs {y1_indices} to controller inputs {u2_indices}")
    logger.debug(f"Mapping controller outputs {y2_indices} to system inputs {u1_indices}")

    # interconnection algorithm
    # TODO: separate function on matrices
    # For now assume NO MATRICES are empty
    len_u1, len_w1 = len(u1_indices), len(w1_indices)
    len_y1, len_z1 = len(y1_indices), len(z1_indices)
    len_u2, len_w2 = len(u2_indices), len(w2_indices)
    len_y2, len_z2 = len(y2_indices), len(z2_indices)

    # TODO: for now, log table
    nrows = A1.shape[0] + len_u1 + len_y1 + n_controller + len_u2 + len_y2
    ncols = A1.shape[1] + len_u1 + len_y1 + n_controller + len_u2 + len_y2
    ndelays1 = hA1.shape[0] + hB1.shape[0] + hC1.shape[0] + hD1.shape[0]
    ndelays2 = hK.shape[0]
    ndelaysA = ndelays1 + ndelays2
    ndelaysB = hB1.shape[0] + hD1.shape[0] 
    ndelaysC = hC1.shape[0] + hD1.shape[0]
    ndelaysD = hD1.shape[0]

    # initialize E, A, hA, B, hB
    E = np.zeros(shape=(nrows, ncols), dtype=E1.dtype)
    A = np.zeros(shape=(nrows, ncols, ndelaysA), dtype=A1.dtype)
    hA = np.zeros(shape=(ndelaysA,), dtype=hA1.dtype)
    B = np.zeros(shape=(nrows, len_w1 + len_w2, ndelaysB), dtype=B1.dtype)
    hB = np.zeros(shape=(ndelaysB,), dtype=hB1.dtype)
    C = np.zeros(shape=(len_z1 + len_z2, ncols, ndelaysC), dtype=C1.dtype)
    hC = np.zeros(shape=(ndelaysC,), dtype=hC1.dtype)
    D = np.zeros(shape=(len_z1 + len_z2, len_w1 + len_w2, ndelaysD), dtype=D1.dtype)
    hD = np.zeros(shape=(ndelaysD,), dtype=hD1.dtype)

    # system
    rows_start1 = 0 # equations
    rows_end1 = rows_start1 + A1.shape[0] + len_y1
    cols_start1 =  0 # variables
    cols_end1 = cols_start1 + A1.shape[1] + len_u1
    concatenate_2x2_by_delays(
        E1, A1,
        B1[np.ix_(range(B1.shape[0]), u1_indices, range(B1.shape[2]))],
        C1[np.ix_(y1_indices, range(C1.shape[1]), range(C1.shape[2]))],
        D1[np.ix_(y1_indices, u1_indices, range(D1.shape[2]))],
        hA1, hB1, hC1, hD1,
        EE = E[rows_start1:rows_end1, cols_start1:cols_end1],
        AA = A[rows_start1:rows_end1, cols_start1:cols_end1, :ndelays1],
        hAA= hA[:ndelays1],
    )

    # place Controller, here, only delays and E2 needs to be placed
    rows_start2 = A1.shape[0] + len_y1 + len_u2 # equations
    rows_end2 = rows_start2 + n_controller + len_y2
    cols_start2 =  A1.shape[1] + len_u1 + len_y1 # variables
    cols_end2 = cols_start2 + n_controller + len_u2
    E[rows_start2:rows_end2,cols_start2:cols_end2] = E2
    hA[ndelays1:] = hK

    # interconnect system and controller
    # here I assume that hA[0], other approach would be to concatenate
    assert hA[0] == 0
    A[A1.shape[0]:A1.shape[0]+len_y1, cols_end1:cols_end1+len_y1,0] = -np.eye(len_y1) # equation 0 = C1 @ x1 + D1 @ u1 - I @ y1
    A[rows_end1:rows_end1+len_y1, cols_end1:cols_end1+len_y1,0] = np.eye(len_y1) # equation 0 = I @ y1 - I @ u2 - PART y1
    A[rows_end1:rows_end1+len_y1, cols_start2+n_controller:cols_start2+n_controller+len_u2,0] = -np.eye(len_u2) # equation 0 = I @ y1 - I @ u2 - PART u2
    A[rows_start2+n_controller:rows_start2+n_controller+len_y2, cols_end2:cols_end2+len_y2,0] = -np.eye(len_y2) # equation 0 = C2 @ x2 + D2 @ u2 - I @ y2
    A[rows_end2:rows_end2+len_y2, cols_end2:cols_end2+len_y2,0] = np.eye(len_y2) # equation 0 = I @ y2 - I @ u1 - PART y2
    A[rows_end2:rows_end2+len_u1, A1.shape[1]:A1.shape[1]+len_u1,0] = -np.eye(len_u1) # equation 0 = I @ y2 - I @ u1 - PART u1

    # Populate B, hB
    n, m = hB1.shape[0], hD1.shape[0]
    B[:B1.shape[0], :len_w1, :n] = B1[np.ix_(range(B1.shape[0]), w1_indices, range(B1.shape[2]))] # B1: w1 -> x1
    B[B1.shape[0]: B1.shape[0]+len_y1, :len_w1, n:n+m] = D1[np.ix_(y1_indices, w1_indices, range(D1.shape[2]))] # D1: w1 -> y1
    
    hB[:n] = hB1
    hB[n:n+m] = hD1

    # Populate C, hC
    n, m = hC1.shape[0], hD1.shape[0]
    C[:len_z1, :C1.shape[1], :n] = C1[np.ix_(z1_indices, range(C1.shape[1]), range(C1.shape[2]))] # C1: x1 -> z1
    C[:len_z1, C1.shape[1]:C1.shape[1]+len_u1, n:n+m] = D1[np.ix_(z1_indices, u1_indices, range(D1.shape[2]))] # D1: u1 -> z1
    
    hC[:n] = hC1
    hC[n:n+m] = hD1

    # Populate D, hD
    # m, q reused from before: m=hD1.shape[0], q=hD2.shape[0]
    D[:len_z1, :len_w1, :m] = D1[np.ix_(z1_indices, w1_indices, range(D1.shape[2]))] # D1: w1 -> z1
    hD[:m] = hD1

    return DDAE(A=A, hA=hA, E=E, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)

def interconnect3(system: DDAE, y_indices: list=None, u_indices:list = None, 
                  hA2: npt.NDArray = None, hB2: npt.NDArray = None, 
                  hC2: npt.NDArray = None, hD2: npt.NDArray = None,
                  **kwargs) -> DDAE:
    """ Creates Closed Loop representation """
    # perform checks

    # all IO matrices defined
    # correct and possible shapes

    # First, handle defaults
    if y_indices is None:
        y_indices = [0]
    if u_indices is None:
        u_indices = [0]
    
    y1_indices = y_indices
    u1_indices = u_indices

    # controller packed into TODO
    n_controller = 1 # order of controller, 0, 1, ..., inf
    n_inputs = len(y1_indices) # assumed to be at least 1
    n_outputs = len(u1_indices) # assumed to be at least 1
    E2, K, hK = controller_reprezentation(order=n_controller, n_inputs=n_inputs, n_outputs=n_outputs)

    nk, mk = n_controller + n_inputs, n_controller + n_inputs

    # Extract matrices and delays of a system
    E1, A1, B1, C1, D1 = system.E, system.A, system.B, system.C, system.D
    hA1, hB1, hC1, hD1 = system.hA, system.hB, system.hC, system.hD

    # TODO argument names and make sure it is list
    # indices all are constructed as 1D arrays and axes are added later to leverage
    # broadcasting, it is necessary to have indices as dtype=int (alternatively as boolean mask)
    
    ## System 1
    u1_indices = np.array(u1_indices, dtype=int)
    w1_indices = np.array([i for i in range(B1.shape[1]) if i not in u1_indices], dtype=int)
    y1_indices = np.array(y1_indices, dtype=int)
    z1_indices = np.array([i for i in range(C1.shape[0]) if i not in y1_indices], dtype=int)

    ## Controller
    u2_indices = np.array([i for i in range(n_inputs)], dtype=int)
    w2_indices = np.array([], dtype=int) # empty by definition
    y2_indices = np.array([i for i in range(n_outputs)], dtype=int)
    z2_indices = np.array([], dtype=int) # empty by definiiton

    # log interconection indices
    logger.debug(f"Mapping system outputs {y1_indices} to controller inputs {u2_indices}")
    logger.debug(f"Mapping controller outputs {y2_indices} to system inputs {u1_indices}")

    # interconnection algorithm
    # TODO: separate function on matrices
    # For now assume NO MATRICES are empty
    len_u1, len_w1 = len(u1_indices), len(w1_indices)
    len_y1, len_z1 = len(y1_indices), len(z1_indices)
    len_u2, len_w2 = len(u2_indices), len(w2_indices)
    len_y2, len_z2 = len(y2_indices), len(z2_indices)

    # TODO: for now, log table
    nrows = A1.shape[0] + len_u1 + len_w1 + len_z1 + n_controller + len_y1
    ncols = A1.shape[1] + len_u1 + len_w1 + len_z1 + n_controller + len_y1
    
    ndelays1 = hA1.shape[0] + hB1.shape[0] + hC1.shape[0] + hD1.shape[0]
    ndelays2 = hK.shape[0]
    ndelaysA = ndelays1 + ndelays2
    ndelaysB = 1
    ndelaysC = 1
    ndelaysD = 1

    # initialize E, A, hA, B, hB
    # we assume the vector of variables to be
    # x* = [x, @u, @w, @z, xc, @y]
    # and equations TODO
    E = np.zeros(shape=(nrows, ncols), dtype=E1.dtype)
    A = np.zeros(shape=(nrows, ncols, ndelaysA), dtype=A1.dtype)
    hA = np.zeros(shape=(ndelaysA,), dtype=hA1.dtype)
    B = np.zeros(shape=(nrows, len_w1 + len_w2, ndelaysB), dtype=B1.dtype)
    hB = np.zeros(shape=(ndelaysB,), dtype=hB1.dtype)
    C = np.zeros(shape=(len_z1 + len_z2, ncols, ndelaysC), dtype=C1.dtype)
    hC = np.zeros(shape=(ndelaysC,), dtype=hC1.dtype)
    D = np.zeros(shape=(len_z1 + len_z2, len_w1 + len_w2, ndelaysD), dtype=D1.dtype)
    hD = np.zeros(shape=(ndelaysD,), dtype=hD1.dtype)

    # system
    rows_start1 = 0 # equations
    rows_end1 = rows_start1 + A1.shape[0] + len_y1 + len_z1
    cols_start1 =  0 # variables
    cols_end1 = cols_start1 + A1.shape[1] + len_u1 + len_w1
    concatenate_2x2_by_delays(
        E1, A1,
        B1[np.ix_(range(B1.shape[0]), np.r_[u1_indices, w1_indices], range(B1.shape[2]))],
        C1[np.ix_(np.r_[y1_indices, z1_indices], range(C1.shape[1]), range(C1.shape[2]))],
        D1[np.ix_(np.r_[y1_indices, z1_indices], np.r_[u1_indices, w1_indices], range(D1.shape[2]))],
        hA1, hB1, hC1, hD1,
        EE = E[rows_start1:rows_end1, cols_start1:cols_end1],
        AA = A[rows_start1:rows_end1, cols_start1:cols_end1, :ndelays1],
        hAA= hA[:ndelays1],
    )

    # interconnect system and controller
    # here I assume that hA[0], other approach would be to concatenate
    assert hA[0] == 0
    n_auxiliary = len_u1 + len_w1 + len_z1 + len_y1 # number of auxiliary variables
    indices = np.array([i for i in range(A1.shape[0], A1.shape[0]+n_auxiliary)], dtype=int)
    A[indices, indices[::-1], 0] = -1

    # BB and CC matrix, such that A[:,:,i] = P[:,:,i] + BB @ K[:,:,i] @ CC
    BB = np.zeros(shape=(nrows, n_controller+len_u1), dtype=np.float64)
    BB[-n_controller-len_u1:,:] = np.eye(n_controller+len_u1)[::-1,:]
    CC = np.zeros(shape=(n_controller+len_y1, ncols), dtype=np.float64)
    CC[:,-n_controller-len_y1:] = np.eye(n_controller+len_y1)[::-1,:]

    # add controller LHS
    E = E + BB @ E2 @ CC # this also tests dimensions

    # prepare B, C, D matrices
    B = np.zeros(shape=(nrows, len_w1, 1), dtype=np.float64)
    B[A1.shape[0] + len_y1 + len_z1:A1.shape[0] + len_y1 + len_z1+len_w1, :, 0] = np.eye(len_w1)
    C = np.zeros(shape=(len_z1, ncols, 1), dtype=np.float64)
    C[:, A1.shape[1] + len_u1 + len_w1:A1.shape[1] + len_u1 + len_w1+len_z1,0] = np.eye(len_z1)
    D = np.zeros(shape=(len_z1, len_w1, 1), dtype=np.float64)

    hB = np.array([0.0])
    hC = np.array([0.0])
    hD = np.array([0.0])

    # # Populate B, hB
    # n, m = hB1.shape[0], hD1.shape[0]
    # B[:B1.shape[0], :len_w1, :n] = B1[np.ix_(range(B1.shape[0]), w1_indices, range(B1.shape[2]))] # B1: w1 -> x1
    # B[B1.shape[0]: B1.shape[0]+len_y1, :len_w1, n:n+m] = D1[np.ix_(y1_indices, w1_indices, range(D1.shape[2]))] # D1: w1 -> y1
    
    # hB[:n] = hB1
    # hB[n:n+m] = hD1

    # # Populate C, hC
    # n, m = hC1.shape[0], hD1.shape[0]
    # C[:len_z1, :C1.shape[1], :n] = C1[np.ix_(z1_indices, range(C1.shape[1]), range(C1.shape[2]))] # C1: x1 -> z1
    # C[:len_z1, C1.shape[1]:C1.shape[1]+len_u1, n:n+m] = D1[np.ix_(z1_indices, u1_indices, range(D1.shape[2]))] # D1: u1 -> z1
    
    # hC[:n] = hC1
    # hC[n:n+m] = hD1

    # # Populate D, hD
    # # m, q reused from before: m=hD1.shape[0], q=hD2.shape[0]
    # D[:len_z1, :len_w1, :m] = D1[np.ix_(z1_indices, w1_indices, range(D1.shape[2]))] # D1: w1 -> z1
    # hD[:m] = hD1

    A, hA = compress_matrices_delays(A, hA)
    B, hB = compress_matrices_delays(B, hB)
    C, hC = compress_matrices_delays(C, hC)
    D, hD = compress_matrices_delays(D, hD)

    return DDAE(A=A, hA=hA, E=E, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD), BB, CC
