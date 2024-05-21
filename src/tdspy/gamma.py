"""
implementation of tds_gamma_r
"""

import logging

from .ddae import DDAE

logger = logging.getLogger()


def gamma(ddae: DDAE, r: float,  **kwargs):
    logger.warning("Gamma R not implemented yet!")
    return 0.5 # TODO
