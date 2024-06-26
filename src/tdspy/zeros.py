"""
Functionalities for calculating transmission zeros
"""

from .rdde import RDDE
from .ddae import DDAE
from .ndde import NDDE
from .roots import roots

def zeros(tds:RDDE | NDDE | DDAE , r, **kwargs):
    """ Computes transmission zeros of a SISO time-delay system

    Args:
        tds (TODO): instance of time-delay system, i.e., RDDE, NDDE or DDAE
        r (int or list): rectangular with 4 coordinates [Re_min, Re_max, Im_min, Im_max]
        kwargs:
            max_size_evp (int): TODO, default 600
            discretization (int): discretization, if None heuristic is envoked,
                default None, keep default if you don't know, has to be > 1
            basic_delay (float): define if delays are commensurate, default
                None, used in discretization heuristic case `rhp`
    
    """

    # perform checks

    # form new DDAE

    # call roots on new DDAE

    raise NotImplementedError("...")