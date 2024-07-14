"""
Functionalities for calculating transmission zeros
"""

import logging

import numpy as np
import numpy.typing as npt

from .rdde import RDDE
from .ddae import DDAE
from .ndde import NDDE
from .common.delay_difference_equation import normalize_diff
from .common.compress import compress_matrices_delays
from .stability.characteristic_roots import roots_ddae, RootsInfo

def zeros(tds: RDDE | NDDE | DDAE, r, input_index: int=0, output_index: int=0, **kwargs):
    """ Computes transmission zeros of a SISO time-delay system

    Args:
        tds (TDS): instance of time-delay system, i.e., RDDE, NDDE or DDAE
        r (list): rectangular region defined via 4 coordinates 
            [Re_min, Re_max, Im_min, Im_max]
        input_index (int): input index for transmission zeros computation,
            default 0
        output_index (int): output index for transmission zeros computation,
        **kwargs:
            max_size_evp (int): TODO, default 600
            discretization (int): discretization, if None heuristic is envoked,
                default None, keep default if you don't know, has to be > 1
            basic_delay (float): define if delays are commensurate, default
                None, used in discretization heuristic case `rhp`
    
    """
    # perform checks TODO
    
    ## validate tds
    
    ## validate region

    ## assert at least one input and one output

    ## validate input output index
    
    # type of TDS, RDDE and DDAE -> OK, NDDE -> convert to DDAE
    if isinstance(tds, NDDE):
        tds = tds.to_ddae()
    
    # form new DDAE
    nrows = tds.A.shape[0] + 1
    ncols = tds.A.shape[1] + 1
    ndelays = tds.mA + tds.mB + tds.mC + tds.mD

    E = np.zeros(shape=(nrows, ncols), dtype=tds.E.dtype)
    A = np.zeros(shape=(nrows, ncols, ndelays), dtype=tds.A.dtype)
    
    hA = np.r_[tds.hA, tds.hB, tds.hC, tds.hD]
    E[:-1, :-1] = tds.E
    A[:-1, :-1, :tds.mA] = tds.A
    A[:-1, -1, tds.mA:tds.mA+tds.mB] = tds.B[:, input_index, :]
    A[-1, :-1, tds.mA+tds.mB: tds.mA+tds.mB+tds.mC] = tds.C[output_index, :, :]
    A[-1, -1, tds.mA+tds.mB+tds.mC:] = tds.D[output_index, input_index, :]

    


    # compress
    compressed_A, compressed_hA = compress_matrices_delays(A, hA)

    print(compressed_hA)
    print(E)
    for i in range(compressed_A.shape[2]):
        print(compressed_A[:,:,i])

    # obtain roots of new (E, A, hA) <=> transmission zeros
    cr, cr_info = roots_ddae(E, compressed_A, compressed_hA, r=r)

    return cr