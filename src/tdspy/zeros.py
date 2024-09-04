"""
Functionalities for calculating transmission zeros
"""

import logging

import numpy as np
import numpy.typing as npt

from .rdde import RDDE
from .ddae import DDAE
from .ndde import NDDE
from .common.compress import compress_matrices_delays
from .common.composition import concatenate_2x2_by_delays
from .stability.characteristic_roots import roots_ddae, RootsInfo

def zeros(tds: RDDE | NDDE | DDAE, r: list, input_index: int=0, output_index: int=0, **kwargs)->tuple[npt.NDArray, RootsInfo]:
    """ Computes transmission zeros of a time-delay system
    
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
    
    Returns:
            tuple containing

                - roots (array): array of found roots
                - metadata (RootsInfo): named tuple consisting of usefull
                    metadata
    """
    
    ## validate region
    assert isinstance(r, (list, tuple, npt.NDArray)), "incorrect region type"
    assert len(r) == 4, "region has to be defined in form [a,b,c,d]"
    assert r[0] < r[1] and r[2] < r[3], "region has to be defined as [a,b,c,d], a<b, c<d"
    assert np.all(~np.isinf(r)), "region has to be finite rectangle"

    ## assert at least one input and one output
    assert tds.n_inputs > 0, "tds has to have at least one input"
    assert tds.n_outputs > 0, "tds has to have at least one output"
    
    ## validate input output index
    assert input_index < tds.n_inputs, "provided input index has to be valid"
    assert output_index < tds.n_outputs, "provided output index has to be valid"
    
    # type of TDS, RDDE and DDAE -> OK, NDDE -> convert to DDAE
    if isinstance(tds, NDDE):
        tds = tds.to_ddae()
    
    # form new DDAE representing transmission zeros problem
    E, A, hA = concatenate_2x2_by_delays(
        tds.E, tds.A, tds.B[:, [input_index], :], tds.C[[output_index], :, :],
        tds.D[[output_index], [input_index], :], tds.hA, tds.hB, tds.hC, tds.hD,
    )        

    # compress - TODO consider as kwarg? or always do compression?
    compressed_A, compressed_hA = compress_matrices_delays(A, hA)

    # roots of new DDAE (E, A, hA) <=> transmission zeros
    cr, cr_info = roots_ddae(E, compressed_A, compressed_hA, r=r, **kwargs)

    return cr, cr_info