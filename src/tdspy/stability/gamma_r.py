"""
Computation of gamma r
----------------------
TODO
"""

import logging
import numpy as np
import numpy.typing as npt

from scipy import linalg

logger = logging.getLogger(__name__)

def compute_gamma_r(DD: list[npt.NDArray], hDD: npt.NDArray, r: float, **kwargs):
    """ TODO

    Args:
        DD (list of `ndarray`)
        hDD (ndarray)
        r (float)

        kwargs:
            n_theta (int): theta discretization, has to be > 0, default 10
            correction (bool): if correction is applied, default True
    """

    n_delays = len(hDD)
    assert n_delays > 0, "empty dde representation not allowed"
    assert n_delays == len(DD), "len of DD and hDD has to match"

    n_theta = kwargs.get("n_theta", 10)
    assert isinstance(n_theta, int), "n_theta has to be of type int"
    assert n_theta > 0, "n_theta has to be > 0"

    if n_theta % 2 == 1:
        n_theta += 1

    if n_delays == 1:
        # CASE 1: 1 delay -> no sensitivity to infinitesimal delay perturbations
        M = DD[0] * np.exp(-r*hDD[0])
        vals = linalg.eig(M)
        gamma_r = np.max(np.abs(vals))
        return gamma_r
    
    # CASE 2: n_delays > 1 in DDE
    ## STEP 1: prediction step -- grid search over [0,2*pi)^{m}
    n_opt = len(DD) - 1 # number of free optimization parameters
    radius = 0 # store maximal value

    id = np.zeros((n_opt,), dtype=int)
    theta_grid = np.linspace(0, 2*np.pi, num=n_theta+1, endpoint=False)

    if all(np.all(np.isreal(d)) for d in DD):
        logger.debug("DDE is real, theta1 can be restricted to [0, pi]")
        endpoint = n_theta // 2 + 1
    else:
        endpoint = n_theta

    while id[0] <= endpoint:
        # the optimization variable theta = [0 theta_grid(id)] -> we do not need to explicitly form the search grid
        # M = DD{1}*exp(-r*hDD(1))*exp(1j*theta(1)) + .. + DD{m}*exp(-r*hDD(m))*exp(1j*theta(m))

        # construct M
        M = DD[0] * np.exp(-r*hDD[0])
        for k2 in range(n_opt):
            M += DD[k2+1] * np.exp(-r*hDD[k2+1]) * np.exp(1j * theta_grid[id[k2]])
        
        vals = linalg.eig(M)        
        gamma_r = np.max(np.abs(vals))
        if gamma_r > radius:
            radius = gamma_r
            #radius_eig = vals[TODO]
            #radius_ind = ind
        
        # form the next gridpoint
        id[-1] += 1
        j = len(id)
        while id[j-1] == n_theta +1:
            if j == 0:
                break
            id[j-1] = 1
            id[j-2] = id[j-2] + 1
            j = j -1

    # Continue line 130





