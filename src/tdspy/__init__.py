import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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
