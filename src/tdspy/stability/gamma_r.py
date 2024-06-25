"""
Computation of gamma(r, TDS)
----------------------------
TODO

Notes:
    1. `MemoizeJac` decorator can be used because of (i) keep implementation as
        simillar as possible to MATLAB® and (ii) to optimize calculation of 
        function value and jacobian as some operations are shared
"""

from collections import namedtuple
import logging
import sys # TODO delete

import numpy as np
import numpy.typing as npt

from scipy import linalg
from scipy import optimize

logger = logging.getLogger(__name__)

GammaInfo = namedtuple("GammaInfo", ["th", "M", "s", "u", "v"])

def func(x: npt.NDArray, DD: npt.NDArray, hDD: npt.NDArray, r, v0):
    """ Calculates value and jacobian of the vector function F(x)

    vector x, shape=(4*ndiff + 2 + (m-1), ):

        x = [Re(v), Im(v), Re(u), Im(u), Re(lambda), Im(s), th]
    
    function F(x):

        M * v - s * v                                                   = 0
        u^{H} * M  - s * u^{H}                                          = 0
        u^{H} * v  - 1                                                  = 0
        v0^{H} * v - 1                                                  = 0
        Im(conj(lambda)*(u^{H}*DD{k}*v)*exp(-r*hDD(k))*exp(1j*theta(k)) = 0 for k = 1,...,m
        
    with th = [0; th_v], v0 a normalization vector and
        
        M = DD[0]*exp(-r*hDD[0])*exp(1j*th[1]) + ... + DD[m]*exp(-r*hDD[m])*exp(1j*th[m])
        
    using fsolve (#optim. variables: 4*ndiff + 2 + (m-1), #constraints: 4*ndiff + 2 + (m-1) ).

    Returns:
        tuple containing

        - y (array): 1D array representing function F evaluated at x
        - jac (array): jacobian of F evaluated at x
    
    Notes:
        1. shape of `x` (n_opt, ), n_opt = 4*ndiff + 2 + (m-1)
        1. shape of `jac` (2*(2*n_diff+2) + n_opt, 2*(2*n_diff+1) + n_opt),
            n_diff ... dimension of the state vector of the delay-difference equation
    """
    n_diff = DD.shape[0] # TODO this could be calculated in advance
    n_opt = DD.shape[2] - 1 # TODO this could be calculated in advance

    # unpack opt. variables x -> u, v, lambda, theta
    v = x[:n_diff] + 1j*x[n_diff:2*n_diff]
    u = x[2*n_diff:3*n_diff] + 1j*x[3*n_diff:4*n_diff]
    s = x[4*n_diff] + 1j*x[4*n_diff+1] # lambda
    th = x[4*n_diff+2:] # theta

    # construct M
    M = DD[:,:,0] * np.exp(-r*hDD[0])
    for i in range(n_opt):
        M = M + DD[:,:,i+1] * np.exp(-r*hDD[i+1]) * np.exp(1j * th[i])

    # continue line 270
    M1 = M - s * np.eye(n_diff)
    M2 = np.conj(M).T - np.conj(s) * np.eye(n_diff)
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
    # y[4*n_diff+4:] = 0 automatically fullfiled

    # construct jacobian
    jac_shape = (2*(2*n_diff+2) + n_opt, 2*(2*n_diff+1) + n_opt)
    jac = np.zeros(shape=jac_shape)

    jac[:n_diff, :n_diff] = np.real(M1)
    jac[:n_diff, n_diff: 2*n_diff] = -np.imag(M1)
    jac[n_diff: 2*n_diff, :n_diff] = np.imag(M1)
    jac[n_diff: 2*n_diff, n_diff: 2*n_diff] = np.real(M1)
    jac[:n_diff, 4*n_diff] = -np.real(v)  # Jac(1:ndiff,4*ndiff+1) = -real(v);
    jac[:n_diff, 4*n_diff+1] = np.imag(v) # Jac(1:ndiff,4*ndiff+2) = imag(v)
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
        v1 = np.ravel(np.conj(u)[np.newaxis, :] @ DD[:,:,k+1] * np.exp(-r*hDD[k+1])) # v1 = u'*DD{k+1}*exp(-r*hDD(k+1));
        v2 = np.ravel(DD[:,:,k+1] @ v[:,np.newaxis] * np.exp(-r*hDD[k+1])) # v2 = DD{k+1}*v*exp(-r*hDD(k+1));
        uDv = np.inner(v1, v) # uDv = (v1*v);
        M3 = 1j*np.exp(1j*th[k])*v2 # M3 = 1j*exp(1j*theta(k))*v2;
        M4 = -1j*np.conj(v1) * np.exp(-1j*th[k]); # M4 = -1j*conj(v1)*exp(-1j*theta(k));
        y[4*n_diff+4+k] = np.imag( np.conj(s) * np.exp(1j*th[k])*uDv) # y(2*(2*ndiff+2)+k)= imag( conj(lambda)*exp(1j*theta(k))*uDv );
        jac[:n_diff, 4*n_diff+1+k+1] = np.real(M3) # Jac(1:ndiff,4*ndiff+2+k) = real(M3);
        jac[n_diff: 2*n_diff, 4*n_diff+1+k+1] = np.imag(M3) # Jac(ndiff+(1:ndiff),4*ndiff+2+k) = imag(M3);
        jac[2*n_diff: 3*n_diff, 4*n_diff+1+k+1] = np.real(M4) # Jac(2*ndiff+(1:ndiff),4*ndiff+2+k) = real(M4);
        jac[3*n_diff: 4*n_diff, 4*n_diff+1+k+1] = np.imag(M4) # Jac(3*ndiff+(1:ndiff),4*ndiff+2+k) = imag(M4);
        jac[4*n_diff+3+k+1, :n_diff] = np.imag(np.conj(s)*np.exp(1j*th[k])*v1) # Jac(4*ndiff+4+k,1:ndiff) = imag(conj(lambda)*exp(1j*theta(k))*v1);
        jac[4*n_diff+3+k+1, n_diff :2*n_diff] = np.real(np.conj(s)*np.exp(1j*th[k])*v1) # Jac(4*ndiff+4+k,ndiff+(1:ndiff)) = real(conj(lambda)*exp(1j*theta(k))*v1);
        jac[4*n_diff+3+k+1, 2*n_diff: 3*n_diff] = np.imag(np.conj(s)*np.exp(1j*th[k])*v2) # Jac(4*ndiff+4+k,2*ndiff+(1:ndiff)) = imag(conj(lambda)*exp(1j*theta(k))*v2);
        jac[4*n_diff+3+k+1, 3*n_diff: 4*n_diff] = -np.real(np.conj(s)*np.exp(1j*th[k])*v2)# Jac(4*ndiff+4+k,3*ndiff+(1:ndiff)) = -real(conj(lambda)*exp(1j*theta(k))*v2);
        jac[4*n_diff+3+k+1, 4*n_diff] = np.imag(np.exp(1j*th[k])*uDv) # Jac(4*ndiff+4+k,4*ndiff+1) = imag(exp(1j*theta(k))*uDv);
        jac[4*n_diff+3+k+1, 4*n_diff+1] =  -np.real(np.exp(1j*th[k])*uDv) # Jac(4*ndiff+4+k,4*ndiff+2) = -real(exp(1j*theta(k))*uDv);
        jac[4*n_diff+3+k+1, 4*n_diff+1+k+1] = np.imag(1j*np.conj(s)*np.exp(1j*th[k])*uDv) # Jac(4*ndiff+4+k,4*ndiff+2+k) = imag(1j*conj(lambda)*exp(1j*theta(k))*uDv);
    
    return y, jac

def l2_func(*args):
    """ returns L2 of F(x) and its jacobian """

    y, jac = func(*args)

    jac_new = np.sum(2 * jac.T * y, axis=1)
    y_new = np.linalg.norm(y)**2

    return y_new, jac_new

def compute_gamma(DD: npt.NDArray, hDD: npt.NDArray, r: float, **kwargs) -> tuple[float, GammaInfo]:
    """ Computes gamma(r) of the normalized delay difference equation
    
    Normalized delay difference equation takes form
        0 = x(t) + DD[0] * x(t-hDD[0]) + ... + DD[m-1]*x(t-hDD[m-1]),      (1)
    where m == len(DD) == len(hDD).

    The gamma(r) of (1) is given by the following optimization problem:
        find maximum theta from [0, 2*pi)^m of expression:
            rho( SUM for all k DD[k]*exp(-r*hDD[k])*exp(1j*theta[k]) )     (2)
        where rho(.) is spectral radius of its matrix argument.

    Args:
        DD (array): coefficient matrices packed into 3D array shaped (n,n,m),
            note that coefficients for x(t) are assumed to be identity matrix
            and therefore omitted (see functions for converting DDAE to delay
            difference equation and normalizing)
        hDD (array): delays represented by 1D array shaped (m,), note that delay
            0 is omitted
        r (float): TODO

        kwargs:
            n_theta (int): theta discretization, has to be > 0, default 10
            correction (bool): if correction is applied, default True
            scipy_root_method (str): scipy.optimize.root method, default 'lm',
                i.e. Levenberg-Marquardt algorithm, note: carefull, not all
                methods attempt to solve problem
            scipy_root_tol (float): Tolerance for termination. For detailed
                control, use `scipy_root_options`, default None
            scipy_root_callback (function): Optional callback function. It is
                called on every iteration as `callback(x, f)` where x is the
                current solution and f the corresponding residual. For all
                methods but 'hybr' and 'lm'.
            scipy_root_options (dict): a dictionary of solver options (method),
                default None
    
    Returns:
        tuple containing:
            gamma (float): TODO
            info (GammaInfo): TODO

    Notes:
        1. for all kwargs starting with 'scipy_*' check the following documentation
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html

    """

    assert hDD.size > 0 and DD.size > 0, "empty delay-difference equation not allowed"
    n_delays = hDD.shape[0]
    assert n_delays > 0, "empty delay difference equation not allowed"
    assert n_delays == DD.shape[2], "len of DD and hDD has to match"

    n_diff = DD.shape[0] # dimension of state vector of delay-diff. eq.

    n_theta = kwargs.get("n_theta", 10)
    assert isinstance(n_theta, int), "n_theta has to be of type int"
    assert n_theta > 0, "n_theta has to be > 0"

    correction = kwargs.get("correction", True) # whether to apply correction

    if n_theta % 2 == 1:
        n_theta += 1

    if n_delays == 1:
        # CASE 1: 1 delay -> no sensitivity to infinitesimal delay perturbations
        M = np.sum(DD * np.exp(-r * hDD), axis=2)
        vals = linalg.eig(M, left=False, right=False)
        vals_abs = np.abs(vals)
        gamma_r_index = np.argmax(vals_abs)
        gamma_r = vals_abs[gamma_r_index]
        eig_v = vals[gamma_r_index]
        U, _, Vh = linalg.svd(M - eig_v*np.eye(n_diff))
        v = np.conj(Vh[-1])
        u = U[:,-1]
        return gamma_r, GammaInfo(0, M, eig_v, u, v)
    
    # CASE 2: n_delays > 1 in DDE
    ## STEP 1: prediction step -- grid search over [0,2*pi)^{m}
    n_opt = DD.shape[2] - 1 # number of free optimization parameters
    radius = 0 # store maximal value

    id = np.zeros((n_opt,), dtype=int)
    theta_grid = np.linspace(0, 2*np.pi, num=n_theta, endpoint=False)

    if all(np.all(np.isreal(d)) for d in DD):
        logger.debug("Delay-difference equation is real, theta1 can be restricted to [0, pi]")
        endpoint = n_theta // 2 + 1
    else:
        endpoint = n_theta

    while id[0] <= endpoint:
        # the optimization variable theta = [0 theta_grid(id)]
        # -> we do not need to explicitly form the search grid
        # M = DD{1}*exp(-r*hDD(1))*exp(1j*theta(1)) + .. + DD{m}*exp(-r*hDD(m))*exp(1j*theta(m))

        # construct M
        # M = np.sum(DD * np.exp(-r * hDD) * np.exp(), axis=2)
        M = DD[:,:,0] * np.exp(-r*hDD[0])
        for k2 in range(n_opt):
            M = M + DD[:,:,k2+1] * np.exp(-r*hDD[k2+1]) * np.exp(1j * theta_grid[id[k2]])
    
        vals = linalg.eig(M, left=False, right=False)
        vals_abs = np.abs(vals)
        gamma_r_index = np.argmax(vals_abs)
        gamma_r = vals_abs[gamma_r_index]
        if gamma_r > radius:
            radius = gamma_r
            radius_eig = vals[gamma_r_index]
            radius_ind = np.copy(id)
        
        # form the next gridpoint
        id[-1] += 1
        j = len(id)
        while id[j-1] == n_theta:
            if j == 0:
                break
            id[j-1] = 0
            id[j-2] = id[j-2] + 1
            j = j -1

    if radius == 0:
        # degenerate case
        gamma_r = 0
        gamma_info = GammaInfo(
            np.zeros(shape=(n_opt+1,)),
            np.zeros(shape=(n_diff, n_diff)),
            0,
            np.zeros(shape=(n_diff,)),
            np.zeros(shape=(n_diff,)),
        )
        return gamma_r, gamma_info
    
    if not correction: # correction==False by user -> no correction applied
        logger.debug(f"No correction")
        th = theta_grid[radius_ind.astype(int)]
        M = DD[:,:,0] * np.exp(-r*hDD[0])
        for i in range(n_opt):
            M = M + DD[:,:,i+1]*np.exp(-r*hDD[i+1])*np.exp(1j*th[i])
        U, _, Vh = linalg.svd(M - radius_eig*np.eye(n_diff))
        v = np.conj(Vh[-1])
        u = U[:,-1]
        gamma_info = GammaInfo(np.r_[0, th], M, radius_eig, u, v)
        return gamma_r, gamma_info
    
    # correction=True -> apply correction
    logger.debug("Applying correction to gamma_r")
    th_v = theta_grid[radius_ind.astype(int)] # critical values of theta
    eig_v = radius_eig # critical eigen value

    # compute the corresponding left and right eigenvectors
    M = DD[:,:,0] * np.exp(-r*hDD[0])
    for i in range(n_opt):
        M = M + DD[:,:,i+1]*np.exp(-r*hDD[i+1])*np.exp(1j*th_v[i])    

    U, _, Vh = linalg.svd(M - eig_v*np.eye(n_diff))
    v_s = np.conj(Vh[-1])
    u_s = U[:,-1]
    # normalize u_s, such that u_s' * v_s == 1
    u_s = u_s / np.conj(np.inner(np.conj(u_s), v_s))

    ## Optimization process
    x0 = np.r_[np.real(v_s), np.imag(v_s), np.real(u_s), np.imag(u_s),
               np.real(eig_v), np.imag(eig_v), th_v]
    # return lambda x: func(x, DD, hDD, r, v_s), x0 # TODO delete

    # solve non-lienear root finding problem, use **kwargs starting 'scipy_root_'    
    sol = optimize.root(
        func,
        x0,
        args=(DD, hDD, r, v_s),
        jac=True,
        method=kwargs.get("scipy_root_method", "lm"),
        tol=kwargs.get("scipy_root_tol", None),
        callback=kwargs.get("scipy_root_callback", None),
        options=kwargs.get("scipy_root_options", None),
    ) # solution is saved in sol.x

    # logger.info(f"Using solver {kwargs.get('scipy_minimize_method', 'trust-constr')}")
    # logger.debug(f"{x0=}")
    # sol = optimize.minimize(
    #     l2_func,
    #     x0,
    #     args=(DD, hDD, r, v_s),
    #     jac=True,
    #     hess=None,
    #     method=kwargs.get("scipy_minimize_method", "trust-constr"),
    #     options={'disp': True},
    # ) # solution is saved in sol.x

    if not sol.success: # i.e. root-finding algorithm failed
        logger.warning("Correction step failed")
        # TODO log these:
        # sol.status #
        # sol.message
        # sol.nit
        # sol.nfev # f evals
        # sol.njev
        # sol.nhev
    
    x_star = sol.x # solution x*
    th_star = x_star[4*n_diff+2:]

    M_star = DD[:,:,0] * np.exp(-r*hDD[0])
    for i in range(n_opt):
        M_star = M_star + DD[:,:,i+1]*np.exp(-r*hDD[i+1])*np.exp(1j*th_star[i])
    
    vals = linalg.eig(M_star, left=False, right=False)
    vals_abs = np.abs(vals)
    gamma_r_index = np.argmax(vals_abs)
    gamma_r = vals_abs[gamma_r_index]
    
    improvement = gamma_r - radius
    if improvement < 0.0: # improvement is worse then 0.0
        logger.debug(f"Correction failed {gamma_r=}, {radius=}, rely on radius (predictor)")
        gamma_info = GammaInfo(np.r_[0, th_v], M, eig_v, u_s, v_s)
        return radius, gamma_info
    else:
        logger.debug(f"Correction succesful {gamma_r=}, {radius=}, {improvement=}, using gamma_r")
        eig_star = vals[gamma_r_index]
        U_star, _, Vh_star = linalg.svd(M - eig_star*np.eye(n_diff))
        v_star = np.conj(Vh_star[-1])
        u_star = U_star[:,-1]
        gamma_info = GammaInfo(np.r_[0, th_star], M_star, eig_star, u_star, v_star)
        return gamma_r, gamma_info
