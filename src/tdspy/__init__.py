import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
 
def init_logger(level=logging.INFO, format: str="%(asctime)s - %(name)s - %(levelname)s - %(message)s") -> logging.Logger:
    """ Initializes tdspy logger with Streamhandler and level """
    logger = logging.getLogger(__name__)
    stream_formatter = logging.Formatter(format)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(level=level)
    return logger

# High-level API
## TDS representation objects
from .ddae import DDAE
from .rdde import RDDE
from .ndde import NDDE
from .closed_loop import ClosedLoop

## Functions (on TDS objects)
from .roots import roots
from .zeros import zeros
from .gamma import gamma
from .spectral_abscissa import (spectral_abscissa, spectral_abscissa_diff, 
                                strong_spectral_abscissa, sa, cd, strong_sa)

## Utils
from . import utils
