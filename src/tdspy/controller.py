"""
Set of high level API functions for creating controllers


TODO:
    1. implementation of low level functions as well (matrices)
"""

import logging

from .rdde import RDDE
from .ndde import NDDE
from .ddae import DDAE


logger = logging.getLogger(__name__)


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


