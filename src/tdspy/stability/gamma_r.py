"""
Computation of gamma r
----------------------
TODO

Notes:
    1. `MemoizeJac` decorator is used because of (i) keep implementation as
        simillar as possible to MATLAB® and (ii) to optimize calculation of 
        function value and jacobian as some operations can be shared
"""

import logging
import numpy as np
import numpy.typing as npt

from scipy import linalg
from scipy.optimize._optimize import MemoizeJac

logger = logging.getLogger(__name__)

def func(x: npt.NDArray, DD: list[npt.NDArray], hDD: npt.NDArray, r, v0):
    """ Calculates value and jacobian of the following function
    
    

    TODO:
        1. possible to have x not as a vector but as a 2d array, could be better computation-wise
    """
    n_diff = np.shape(DD[0])[0] # TODO this could be calculated in advance
    n_opt = len(DD) - 1 # TODO this could be calculated in advance

    # unpack opt. variables x -> u, v, lambda, theta
    v = x[:n_diff] + 1j*x[n_diff:2*n_diff]
    u = x[2*n_diff:3*n_diff] + 1j*x[3*n_diff:4*n_diff]
    s = x[4*n_diff] + 1j*x[4*n_diff+1] # lambda
    th = x[4*n_diff+2:] # theta

    # construct M
    M = DD[0] * np.exp(-r*hDD[0])
    for i in range(n_opt):
        M += DD[i+1] * np.exp(-r*hDD[i+1]) * np.exp(1j * th[i])

    # continue line 270
    M1 = M - s * np.eye(n_diff)
    M2 = M.H - np.conj(s) * np.eye(n_diff)
    block_1 = np.ravel(M1 @ v[:,np.newaxis]) # (n_opt, 1)
    block_2 = np.ravel(M2 @ u[:,np.newaxis]) # (n_opt, 1)
    block_3 = np.inner(np.conj(v0), v) - 1
    block_4 = np.inner(np.conj(u), v) - 1

    # construct y = f(x)
    #y=[real(block1); imag(block1); real(block2); imag(block2); real(block3);
    #   imag(block3); real(block4);imag(block4);zeros(n_opt,1)];
    y = np.zeros(shape=(4*n_diff+4+n_opt,))
    y[:n_diff] = np.real(block_1)
    y[n_diff: 2*n_diff] = np.imag(block_1)
    y[2*n_diff: 3*n_diff] = np.real(block_2)
    y[3*n_diff: 4*n_diff] = np.imag(block_2)
    y[4*n_diff] = np.real(block_3)
    y[4*n_diff+1] = np.imag(block_3)
    y[4*n_diff+2] = np.real(block_4)
    y[4*n_diff+3] = np.imag(block_4)
    # y[4*n_diff+4:] = 0 automatically

    # construct jacobian
    jac_shape = (2*(2*n_diff+2) + n_opt, 2*(2*n_diff+1) + n_opt)
    jac = np.zeros(shape=jac_shape)

    jac[:n_diff, :n_diff] = np.real(M1)
    jac[:n_diff, n_diff: 2*n_diff] = -np.imag(M1)
    jac[n_diff: 2*n_diff, :n_diff] = np.imag(M1)
    jac[n_diff: 2*n_diff, n_diff: 2*n_diff] = np.real(M1)
    jac[:n_diff, 4*n_diff] = -np.real(v) # Jac(1:ndiff,4*ndiff+1) = -real(v);
    jac[:n_diff, 4*n_diff+1] = np.real(v) # Jac(1:ndiff,4*ndiff+2) = imag(v)
    jac[n_diff: 2*n_diff, 4*n_diff] = -np.imag(v) # Jac(ndiff+(1:ndiff),4*ndiff+1) = -imag(v);
    jac[n_diff: 2*n_diff, 4*n_diff+1] = -np.real(v) # Jac(ndiff+(1:ndiff),4*ndiff+2) = -real(v);
    jac[2*n_diff: 3*n_diff, 2*n_diff: 3*n_diff] = np.real(M2) # Jac(2*ndiff+(1:ndiff),2*ndiff+(1:ndiff)) = real(M2);
    jac[2*n_diff: 3*n_diff, 3*n_diff: 4*n_diff] = -np.imag(M2) # Jac(2*ndiff+(1:ndiff),3*ndiff+(1:ndiff)) = -imag(M2);
    jac[3*n_diff: 4*n_diff, 2*n_diff: 3*n_diff] = np.imag(M2) # Jac(3*ndiff+(1:ndiff),2*ndiff+(1:ndiff)) = imag(M2);
    jac[3*n_diff: 4*n_diff, 3*n_diff: 4*n_diff] = np.real(M2) # Jac(3*ndiff+(1:ndiff),3*ndiff+(1:ndiff)) = real(M2);
    jac[2*n_diff: 3*n_diff, 4*n_diff] = -np.real(u) # Jac(2*ndiff+(1:ndiff),4*ndiff+1) = -real(u);
    jac[2*n_diff: 3*n_diff, 4*n_diff+1] = -np.imag(u) # Jac(2*ndiff+(1:ndiff),4*ndiff+2) = -imag(u);
    jac[3*n_diff: 4*n_diff, 4*n_diff] = -np.imag(u) # Jac(3*ndiff+(1:ndiff),4*ndiff+1) = -imag(u);
    jac[3*n_diff: 4*n_diff, 4*n_diff+1] = np.real(u) # Jac(3*ndiff+(1:ndiff),4*ndiff+2) = real(u);
    jac[4*n_diff, :n_diff] = np.real(v0) # Jac(4*ndiff+1,1:ndiff) = real(v0);
    jac[4*n_diff, n_diff: 2*n_diff] = np.imag(v0) # Jac(4*ndiff+1,ndiff+(1:ndiff)) = imag(v0);
    jac[4*n_diff+1, :n_diff] = -np.imag(v0) # Jac(4*ndiff+2,1:ndiff) = -imag(v0);
    jac[4*n_diff+1, n_diff: 2*n_diff] = np.real(v0) # Jac(4*ndiff+2,ndiff+(1:ndiff)) = real(v0);
    jac[4*n_diff+2, :n_diff] = np.real(u) # Jac(4*ndiff+3,1:ndiff) = real(u);
    jac[4*n_diff+2, n_diff: 2*n_diff] = np.imag(u) # Jac(4*ndiff+3,ndiff+(1:ndiff)) = imag(u);
    jac[4*n_diff+2, 2*n_diff: 3*n_diff] = np.real(v) # Jac(4*ndiff+3,2*ndiff+(1:ndiff)) = real(v);
    jac[4*n_diff+2, n_diff: 2*n_diff] = np.imag(v) # Jac(4*ndiff+3,3*ndiff+(1:ndiff)) = imag(v);
    jac[4*n_diff+3, :n_diff] = -np.imag(u) # Jac(4*ndiff+4,1:ndiff) = -imag(u);
    jac[4*n_diff+3, n_diff :2*n_diff] = np.real(u) # Jac(4*ndiff+4,ndiff+(1:ndiff)) = real(u);
    jac[4*n_diff+3, 2*n_diff: 3*n_diff] = np.imag(v) # Jac(4*ndiff+4,2*ndiff+(1:ndiff)) = imag(v);
    jac[4*n_diff+3, 3*n_diff: 4*n_diff] = -np.real(v) # Jac(4*ndiff+4,3*ndiff+(1:ndiff)) = -real(v);

    # n_opt update
    for k in range(n_opt):
        ...
        # v1 = u'*DD{k+1}*exp(-r*hDD(k+1));
        # v2 = DD{k+1}*v*exp(-r*hDD(k+1));
        # uDv = (v1*v);
        # M3 = 1j*exp(1j*theta(k))*v2;
        # M4 = -1j*conj(v1)*exp(-1j*theta(k));
        # y(2*(2*ndiff+2)+k)= imag( conj(lambda)*exp(1j*theta(k))*uDv ) ;
        # Jac(1:ndiff,4*ndiff+2+k) = real(M3);
        # Jac(ndiff+(1:ndiff),4*ndiff+2+k) = imag(M3);
        # Jac(2*ndiff+(1:ndiff),4*ndiff+2+k) = real(M4);
        # Jac(3*ndiff+(1:ndiff),4*ndiff+2+k) = imag(M4);
        # Jac(4*ndiff+4+k,1:ndiff) = imag(conj(lambda)*exp(1j*theta(k))*v1);
        # Jac(4*ndiff+4+k,ndiff+(1:ndiff)) = real(conj(lambda)*exp(1j*theta(k))*v1);
        # Jac(4*ndiff+4+k,2*ndiff+(1:ndiff)) = imag(conj(lambda)*exp(1j*theta(k))*v2);
        # Jac(4*ndiff+4+k,3*ndiff+(1:ndiff)) = -real(conj(lambda)*exp(1j*theta(k))*v2);
        # Jac(4*ndiff+4+k,4*ndiff+1) = imag(exp(1j*theta(k))*uDv);
        # Jac(4*ndiff+4+k,4*ndiff+2) = -real(exp(1j*theta(k))*uDv);
        # Jac(4*ndiff+4+k,4*ndiff+2+k) = imag(1j*conj(lambda)*exp(1j*theta(k))*uDv);

    return y, jac



def compute_gamma_r(DD: list[npt.NDArray], hDD: npt.NDArray, r: float, **kwargs):
    """ Computes gamma(r) of the normalized delay difference equation
    
    Equation takes form
        0 = x(t) + DD[0] * x(t-hDD[0]) + ... + DD[m-1]*x(t-hDD[m-1]), (1)
    where m == len(DD) == len(hDD).

    The gamma(r) of (1) is given by:
        max_{theta\in[0,2*pi)^{m}} rho(sum_{k} DD{k}*exp(-r*hDD(k))*exp(1j*theta(k)), (2)
    where function rho(.) returns the spectral radius of its matrix argument.

    Args: TODO
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

    n_diff = np.shape(DD[0])[0] # dimension of state vector of delay-diff. eq.

    n_theta = kwargs.get("n_theta", 10)
    assert isinstance(n_theta, int), "n_theta has to be of type int"
    assert n_theta > 0, "n_theta has to be > 0"

    correction: bool = kwargs.get("correction", True) # whether to apply correction

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
        # gamma_r = np.max(np.abs(vals))
        vals_abs = np.abs(vals)
        gamma_r_index = np.argmax(vals_abs)
        gamma_r = vals_abs[gamma_r_index]
        if gamma_r > radius:
            radius = gamma_r
            radius_eig = vals[gamma_r_index]
            radius_ind = id
        
        # form the next gridpoint
        id[-1] += 1
        j = len(id)
        while id[j-1] == n_theta +1:
            if j == 0:
                break
            id[j-1] = 1
            id[j-2] = id[j-2] + 1
            j = j -1

    if radius == 0:
        # degenerate case, TODO return also metadata
        gamma_r = 0
        return gamma_r
    
    if not correction: # correction=False by user -> no correction applied, return
        logger.debug(f"No correction")
        # TODO return metadata
        return gamma_r
    
    # correction=True -> apply correction
    logger.debug("Applying correction to gamma_r")
    th_v = theta_grid[radius_ind.astype(bool)] # critical values of theta
    eig_v = radius_eig # critical eigen value

    # compute the corresponding left and right eigenvectors
    # line 175
    M = DD[0]*np.exp(-r*hDD[0])
    
    for i in range(n_opt):
        M += DD[i+1]*np.exp(-r*hDD[i+1])*np.exp(1j*th_v[i])

    U, _, Vh = linalg.svd(M - eig_v*np.eye(n_diff))
    v_s = Vh[:, -1]
    u_s = U[:, -1]
    # normalize u_s, such that u_s' * v_s == 1
    u_s = u_s / np.conj(np.inner(np.conj(u_s), v_s))

    ## Optimization process
    x0 = np.r_[np.real(v_s), np.imag(v_s), np.real(u_s), np.imag(u_s),
               np.real(eig_v), np.imag(eig_v), th_v]

    logger.info(f"{x0=}")

    # TODO
