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


logger = logging.getLogger(__name__)


def interconnect(tds1: DDAE, tds2: DDAE, ) -> DDAE:
    """ Creates and interconnected system
            _______
        ----> |       |
            | TDS 1 |
            |_______|
    
    TODO create scheme and maybe even equations

    Args:
        tds1 (TDS): system 1 to be interconnected
        tds2 (TDS): system 2 to be interconnected

    Returns:
        interconnected system (DDAE)
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
    ## System 1
    u1_indices = np.array([0], dtype=int)# THIS IS INPUT - TODO arg
    w1_indices = np.array([i for i in range(B1.shape[1]) if i not in u1_indices], dtype=int)
    y1_indices = np.array([0,1,2,3,4,5], dtype=int) # THIS IS INPUT - TODO arg
    z1_indices = np.array([i for i in range(D1.shape[0]) if i not in y1_indices], dtype=int)

    ## System 2
    u2_indices = np.array([0,1,2,3,4,5], dtype=int) # THIS IS INPUT - TODO arg
    w2_indices = np.array([i for i in range(B2.shape[1]) if i not in u2_indices], dtype=int)
    y2_indices = np.array([0], dtype=int) # THIS IS INPUT - TODO arg
    z2_indices = np.array([i for i in range(D2.shape[0]) if i not in y2_indices], dtype=int)

    # TODO log interconection indices
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
    C[len_z1:, cols_start1:cols_start1+C2.shape[1], n+m:n+m+p] = C1[np.ix_(z2_indices, range(C2.shape[1]), range(C2.shape[2]))] # C2: x2 -> z2
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

    # TODO compressions ???
    A, hA = compress_matrices_delays(A, hA)
    B, hB = compress_matrices_delays(B, hB)
    C, hC = compress_matrices_delays(C, hC)
    D, hD = compress_matrices_delays(D, hD)

    np.set_printoptions(suppress=True, linewidth=100000)
    print(f"E")
    print("----------------------------")
    print(E)
    
    for i in range(A.shape[2]):
        print(f"A[:,:,{i}] = delay={hA[i]}")
        print("----------------------------")
        print(A[:,:,i])

    for i in range(B.shape[2]):
        print(f"B[:,:,{i}] = delay={hB[i]}")
        print("----------------------------")
        print(B[:,:,i])
    
    for i in range(C.shape[2]):
        print(f"C[:,:,{i}] = delay={hC[i]}")
        print("----------------------------")
        print(C[:,:,i])
    
    for i in range(D.shape[2]):
        print(f"D[:,:,{i}] = delay={hD[i]}")
        print("----------------------------")
        print(D[:,:,i])

    return DDAE(A=A, hA=hA, E=E, B=B, hB=hB, C=C, hC=hC, D=D, hD=hD)

def create_closed_loop(plant: DDAE | RDDE | NDDE, controller: RDDE, **kwargs):
    """ Creates new TDS object representing closed-loop interconnection of the
    provided plant and controller
    
    Args:
        plant
        controller

        **kwargs

    Returns
        tuple containing:
            - closed_loop
            - closed_loop_metadata

    """

    compress: bool = kwargs.get("compress", True)


    # TODO assert statemets

    # 

    logger.debug(f"Plant: num inputs = {plant.n_iputs}, num outputs = {plant.n_outputs}")
    logger.debug(f"Controller: num inputs = {controller.n_iputs}, num outputs = {controller.n_outputs}")


