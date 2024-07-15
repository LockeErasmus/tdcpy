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
    # indices
    ## System 1
    u1_indices = [0] # THIS IS INPUT - TODO arg
    w1_indices = [i for i in range(B1.shape[1]) if i not in u1_indices]
    y1_indices = [0] # THIS IS INPUT - TODO arg
    z1_indices = [i for i in range(D1.shape[0]) if i not in y1_indices]

    ## System 2
    u2_indices = [0] # THIS IS INPUT - TODO arg
    w2_indices = [i for i in range(B2.shape[1]) if i not in u2_indices]
    y2_indices = [0] # THIS IS INPUT - TODO arg
    z2_indices = [i for i in range(D2.shape[0]) if i not in y2_indices]

    # TODO log interconection indices
    logger.debug(f"Mapping TDS1 outputs {3} to TDS2 inputs {3}")

    # interconnection algorithm
    # TODO: separate function on matrices
    # For now assume NO MATRICES are empty
    len_u1, len_w1 = len(u1_indices), len(w1_indices)
    len_y1, len_z1 = len(y1_indices), len(z1_indices)
    len_u2, len_w2 = len(u2_indices), len(w2_indices)
    len_y2, len_z2 = len(y2_indices), len(z2_indices)
    
    nrows = A1.shape[0] + len_u1 + len_y1 + A2.shape[0] + len_u2 + len_y2
    ncols = A1.shape[1] + len_u1 + len_y1 + A2.shape[1] + len_u2 + len_y2

    # E matrix
    E = np.zeros(shape=(nrows, ncols), dtype=E1.dtype)
    E[:E1.shape[0], :E1.shape[1]] = E1
    E2_start_rows = A1.shape[0] + len_u1 + len_y1
    E2_start_cols = A1.shape[1] + len_u1 + len_y1
    E[E2_start_rows : E2_start_rows + E2.shape[0], E2_start_cols : E2_start_cols+E2.shape[1]] = E2

    # hA delays, A matrix
    # ndelaysA = sum(block.shape[2] for block in (A1, B1, C1, D1, A2, B2, C2, D2)) # TODO DELETE
    hA = np.concatenate([hA1, hB1, hC1, hD1, hA2, hB2, hC2, hD2], axis=0)

    A = np.zeros(shape=(nrows, ncols, hA.shape[0]), dtype=A1.dtype)
    
    
    blocks1 = (A1, B1[:,u1_indices,:], C1[y1_indices,:,:], D1[y1_indices,u1_indices,:]) # A1, B1u, C1y, D1u->y
    blocks2 = (A2, B2[:,u2_indices,:], C2[y2_indices,:,:], D2[y2_indices, u2_indices,:]) # A2, B2u, C2y, D2u->y
    blocks1_indices = ((0, A1.shape[0], 0, A1.shape[1]), # A1
                      (0, B1.shape[0], A1.shape[1], A1.shape[1]+B1.shape[1]), # B1
                      (A1.shape[0], A1.shape[0]+C1.shape[0], 0, C1.shape[1]), # C1
                      (A1.shape[0], A1.shape[0]+C1.shape[0], 0, C1.shape[1])) # D1
                      
    
    start_index = 0
    end_index = None



    for block, indices in blocks, blocks_indices:
        # TODO if block.shape[2] == 0

        rs, re, cs, ce = indices # unpack indices
        end_index = start_index + block.shape[2] # obtain correct end
        A[rs:re, cs:ce, start_index : end_index] = block # put block
        start_index = end_index # update start
        pass


    




    




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


