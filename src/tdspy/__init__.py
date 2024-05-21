import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from .ddae import DDAE
from .rdde import RDDE
from .ndde import NDDE

from .roots import roots
from .gamma import gamma