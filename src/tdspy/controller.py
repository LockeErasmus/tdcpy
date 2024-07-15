"""
Set of high level API functions for creating controllers


TODO:
    1. implementation of low level functions as well (matrices)
    2. as of now, DDAE later make it work for all three types of TDS
"""

import logging

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

    # TODO argument names
    tds1_outputs = 0
    tds2_inputs = 0

    tds1_inputs = 0
    tds2_outputs = 0

    logger.debug(f"Mapping TDS1 outputs {3} to TDS2 inputs {3}")


    # Extract matrices and delays
    E1, A1, B1, C1, D1 = tds1.E, tds1.A, tds1.B, tds1.C, tds1.D
    E2, A2, B2, C2, D2 = tds2.E, tds2.A, tds2.B, tds2.C, tds2.D


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


