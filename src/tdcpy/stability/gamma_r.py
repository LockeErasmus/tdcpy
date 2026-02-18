# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

"""
Computation of gamma(r, DIFF)
-----------------------------

DIFF - delay difference equation represented via matrices (3D array)
and delays (1D array)

TODO:
    ---

Notes:
    1. `MemoizeJac` decorator can be used because of (i) keep implementation as
        simillar as possible to MATLAB® and (ii) to optimize calculation of 
        function value and jacobian as some operations are shared
"""

from collections import namedtuple
import logging

import numpy as np
import numpy.typing as npt

from scipy import linalg
from scipy import optimize

from tdcpy.common.delay_difference_equation import normalize_diff
from tdcpy.common.compress import compress_matrices_delays

logger = logging.getLogger(__name__)

GammaInfo = namedtuple("GammaInfo", ["th", "M", "s", "u", "v"])

def rho_normalized_diff(DD: npt.NDArray, hDD: npt.NDArray, r: float, theta: npt.NDArray) -> npt.NDArray:
    """ Calculates spectral radius of matrix M

    Matrix M is given as sum of rotations of normalized delay difference
    equation.

    Parameters
    ----------
    DD : ndarray
        Coefficient matrices packed into 3D array of shape (n, n, m).
        Note that coefficients for x(t) are assumed to be identity matrix
        and therefore omitted (see functions for converting DDAE to delay
        difference equation and normalizing).
    hDD : ndarray
        Delays represented by 1D array of shape (m,). Note that delay 0 is
        omitted.
    r : float
        Point from complex plane.
    theta : ndarray
        Non-empty 1d array of rotations of shape (m-1,) in [0, 2*pi).

    Returns
    -------
    ndarray
        Spectral radius matrix.

    Notes
    -----
    Does not check for valid inputs.

    """
    M = (DD[:,:, 0] * np.exp(-r*hDD[0]) # first therm rotation is fixed to 0
         + np.sum(DD[:,:, 1:] * np.exp(-r * hDD[1:]) * np.exp(1j*theta), axis=2))
    return M

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

    Parameters
    ----------
    x : ndarray
        1D array of shape (4*ndiff + 2 + (m-1), ) representing optimization variables
    DD : ndarray
        Coefficient matrices packed into 3D array of shape (n, n, m).
    hDD : ndarray
        Delays represented by 1D array of shape (m,).
    r : float
        Point in complex plane.
    v0 : ndarray
        Normalization vector of shape (n, ).
        
    Returns
    -------
        tuple containing
        y : ndarray
            1D array representing function F evaluated at x
        jac : ndarray
            Jacobian of F evaluated at x, shape (2*(2*ndiff+2) + n_opt, 2*(2*ndiff+1) + n_opt)
    
    Notes
    -----

    1.  shape of `x` (n_opt, ), n_opt = 4*ndiff + 2 + (m-1)
    2.  shape of `jac` (2*(2*n_diff+2) + n_opt, 2*(2*n_diff+1) + n_opt),
        n_diff ... dimension of the state vector of the delay-difference equation
        
    """
    n_diff = DD.shape[0] 
    n_opt = DD.shape[2] - 1

    # unpack opt. variables x -> u, v, lambda, theta
    v = x[:n_diff] + 1j*x[n_diff:2*n_diff]
    u = x[2*n_diff:3*n_diff] + 1j*x[3*n_diff:4*n_diff]
    s = x[4*n_diff] + 1j*x[4*n_diff+1] # lambda
    th = x[4*n_diff+2:] # theta

    # construct M
    M = rho_normalized_diff(DD, hDD, r, th)

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

def theta_generator(n_opt: int, n_theta:int=10, is_real: bool=False):
    """ Generator for theta grid points in [0, 2*pi)^{m} """
    # make sure that n_theta is even
    if n_theta % 2 == 1:
        n_theta += 1
    
    # theta_1 can be restricted to [0, pi] for real matrices
    endpoint = n_theta // 2 + 1 if is_real else n_theta

    # create grid, preallocate vectors for indices and theta
    theta_grid = np.linspace(0, 2*np.pi, num=n_theta, endpoint=False)
    id = np.zeros((n_opt,), dtype=int)
    theta = np.zeros((n_opt,), dtype=np.float64)
    
    while id[0] <= endpoint - 1:
        theta[:] = theta_grid[id]
        yield theta    
        # form the next gridpoint
        id[-1] = id[-1] + 1
        for j in range(n_opt-1, 0, -1):
            if id[j] < n_theta:
                break
            else:
                id[j] = 0
                id[j-1] += 1

def gamma_normalized_diff(DD: npt.NDArray, hDD: npt.NDArray, r: float, **kwargs) -> tuple[float, GammaInfo]:
    """ Computes gamma(r) of the normalized delay difference equation
    
    Normalized delay difference equation takes form
        0 = x(t) + DD[0] * x(t-hDD[0]) + ... + DD[m-1]*x(t-hDD[m-1]),      (1)
    where m == len(DD) == len(hDD).

    The gamma(r) of (1) is given by the following optimization problem:
        find maximum theta from [0, 2*pi)^m of expression:
            rho( SUM for all k DD[k]*exp(-r*hDD[k])*exp(1j*theta[k]) )     (2)
        where rho(.) is spectral radius of its matrix argument.

    Parameters
    ----------

    DD : ndarray
        Coefficient matrices packed into 3D array shaped (n,n,m),
        note that coefficients for x(t) are assumed to be identity matrix
        and therefore omitted (see functions for converting DDAE to delay
        difference equation and normalizing).
    hDD : ndarray
        Delays represented by 1D array shaped (m,), note that delay 
        0 is omitted.
    r : float
        Point from complex plane.
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
    
    Returns
    -------
    tuple containing:
            
        - gamma (float): quantity gamma(r, DD, hDD)
        - info (GammaInfo): gamma metadata

    Notes
    -----
    1. for all kwargs starting with 'scipy_*' check the following documentation
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html
    2. uses grid search + optional correction via scipy.optimize.root

    Examples
    --------
    >>> import numpy as np
    >>> from tdcpy.stability.gamma_r import gamma_normalized_diff
    >>> DD = np.zeros((2,2,2), dtype=complex)
    >>> DD[:,:,0] = np.array([[0.7, 0], [0, 0.7]])
    >>> DD[:,:,1] = np.array([[0.1, 0], [0, 0.1]])
    >>> hDD = np.array([1.0, 2.0])
    >>> r = 0.5 + 0.5j
    >>> gamma, info = gamma_normalized_diff(DD, hDD, r, n_theta=5, correction=True)
    >>> print(f"gamma: {gamma}")
    gamma: 0.4613594059159876
    >>> print(f"theta: {info.th}")
    theta: [0.  0.5]
    >>> print(f"dominant eigenvalue: {info.s}")
    dominant eigenvalue: (0.40488096939597285-0.22118748167138752j)
    >>> print(f"right eigenvector: {info.v}")
    right eigenvector: [0.-0.j 1.-0.j]
    >>> print(f"left eigenvector: {info.u}")
    left eigenvector: [ 0.        +0.j         -0.68163876-0.73168887j]

    """

    assert hDD.size > 0 and DD.size > 0, "empty delay-difference equation not allowed"
    assert hDD.shape[0] > 0, "empty delay difference equation not allowed"
    assert hDD.shape[0] == DD.shape[2], "len of DD and hDD has to match"

    n_diff = DD.shape[0] # dimension of state vector of normalized DIFF

    n_theta = kwargs.get("n_theta", 10)
    assert isinstance(n_theta, int), "n_theta has to be of type int"
    assert n_theta > 0, "n_theta has to be > 0"

    correction = kwargs.get("correction", True) # whether to apply correction

    # CASE 1: number of delays equal to 1 in normalized DIFF
    # this means no sensitivity to infinitesimal delay perturbations and
    # theta would be therefore empty vector
    if hDD.shape[0] == 1:
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
    
    # CASE 2: number of delays greater than 1 in normalized DIFF
    ## STEP 1: prediction step -- grid search over [0,2*pi)^{m}
    n_opt = DD.shape[2] - 1 # number of free optimization parameters
    radius = 0 # store maximal value

    # iterate over cartesian product of theta grid [0, 2*pi)^{n_opt}]
    for theta in theta_generator(n_opt, n_theta, is_real=all(np.all(np.isreal(d)) for d in DD)):
        # construct M
        M = rho_normalized_diff(DD, hDD, r, theta)
        vals = linalg.eig(M, left=False, right=False)
        vals_abs = np.abs(vals)
        gamma_r_index = np.argmax(vals_abs) # index of dominant eigenvalue
        gamma_r = vals_abs[gamma_r_index]
        
        if gamma_r > radius:
            radius = gamma_r
            radius_eig = vals[gamma_r_index]
            radius_th = np.copy(theta)

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
        th = radius_th
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
    th_v = radius_th # critical values of theta
    eig_v = radius_eig # critical eigen value

    # compute the corresponding left and right eigenvectors
    M = rho_normalized_diff(DD, hDD, r, th_v)

    U, _, Vh = linalg.svd(M - eig_v*np.eye(n_diff))
    v_s = np.conj(Vh[-1])
    u_s = U[:,-1]
    # normalize u_s, such that u_s' * v_s == 1
    u_s = u_s / np.conj(np.inner(np.conj(u_s), v_s))

    ## Optimization process
    x0 = np.r_[np.real(v_s), np.imag(v_s), np.real(u_s), np.imag(u_s),
               np.real(eig_v), np.imag(eig_v), th_v]

    # solve non-lienear root finding problem, use **kwargs starting 'scipy_root_
    scipy_root_kwargs = {
        "method": kwargs.get("scipy_root_method", "lm"),
        "tol": kwargs.get("scipy_root_tol", None),  
        "callback": kwargs.get("scipy_root_callback", None),
        "options": kwargs.get("scipy_root_options", None),
    }
    logger.debug(f"Applying corrector: `scipy.optimize.root` with settings: {scipy_root_kwargs}")
    sol = optimize.root(
        func,
        x0,
        args=(DD, hDD, r, v_s),
        jac=True,
        **scipy_root_kwargs,
    ) # solution is saved in sol.x
    logger.debug(f"Corrector fnished, succesfull?={sol.success}, status={sol.status}, message={sol.message}")

    if not sol.success: # i.e. root-finding algorithm failed -> rely on predictor
        logger.warning("Correction failed (solver failed), rely on predictor")
        gamma_info = GammaInfo(np.r_[0, th_v], M, eig_v, u_s, v_s)
        return radius, gamma_info
    
    # obtain solution from `sol` and reconstruct metadata
    x_star = sol.x # solution x*
    th_star = x_star[4*n_diff+2:] # we are only interested in theta*
    M_star = rho_normalized_diff(DD, hDD, r, th_star)
    vals = linalg.eig(M_star, left=False, right=False)
    vals_abs = np.abs(vals)
    gamma_r_index = np.argmax(vals_abs)
    gamma_r = vals_abs[gamma_r_index]
    
    # check for improvement
    improvement = gamma_r - radius
    if improvement < 0.0: # improvement is worse then 0.0
        logger.debug(f"Correction failed (negative improvement), rely on predictor. (predictor={radius}, corrector={gamma_r}, {improvement=})")
        gamma_info = GammaInfo(np.r_[0, th_v], M, eig_v, u_s, v_s)
        return radius, gamma_info
    else:
        logger.debug(f"Correction succesful (positive improvement), rely on corrector. (predictor={radius}, corrector={gamma_r}, {improvement=})")
        eig_star = vals[gamma_r_index]
        U_star, _, Vh_star = linalg.svd(M - eig_star*np.eye(n_diff))
        v_star = np.conj(Vh_star[-1])
        u_star = U_star[:,-1]
        gamma_info = GammaInfo(np.r_[0, th_star], M_star, eig_star, u_star, v_star)
        return gamma_r, gamma_info


def gamma_diff(D: npt.NDArray, hD: npt.NDArray, r: float, **kwargs) -> tuple[float, GammaInfo]:
    """ Computes gamma(r) of delay difference equation (DIFF)

    Delay difference equation takes form
        0 = D[0]*x(t) + D[1] * x(t-hD[1]) + ... + DD[m-1]*x(t-hDD[m-1]),   (1)
    where m == len(DD) == len(hDD).

    quantity gamma(r; D, hD) is then:
    1.  gamma(r; D, hD) = 0 IF number of delays (vector hD) is less then 2
    2.  obtained via predictor corrector approach, i.e.
        2a. normalize DIFF (multiply equation (1) by inverse of D[0]) and
            omit first delay = 0 and first normalized matrix = identity
        2b. call `gamma_diff_normalized`
    
    Parameters
    ----------

    D : array
        Coefficient matrices packed into 3D array shaped (n, n, m).
    hD : array
        Delays represented by 1D array shaped (m,).
    r : float
        Point from complex plane.
    kwargs:
        kwargs passed into `gamma_diff_normalized` function.
    
    Returns
    -------

    tuple containing:

        - gamma (float): quantity gamma(r, D, hD)
        - info (GammaInfo): gamma metadata
    
    Notes
    -----

    1.  if compressed version of DIFF contains 2 or more delays,
        invertibility of D[0] is assumed.
    2.  DIFF representation (D, hD) can be emtpy, result will be
        gamma(r; D, hD) = 0.0. Test for emptyness is hD.size == 0.
    3.  for r = 0.0, quantity gamma(r; D, hD) DOES NOT depend on the delays,
        see implementation of `gamma_diff_normalized`
    
    Examples
    --------

    >>> import numpy as np
    >>> from tdcpy.stability.gamma_r import gamma_diff
    >>> D = np.zeros(shape=(2,2,3))
    >>> D[:,:,0] = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> D[:,:,1] = np.array([[0.5, 0.0], [0.0, 0.5]])
    >>> D[:,:,2] = np.array([[0.2, 0.0], [0.0, 0.2]])
    >>> hD = np.array([0.0, 1.0, 2.0])
    >>> r = 0.0
    >>> gamma_val, gamma_info = gamma_diff(D, hD, r)
    >>> print(gamma_val)
    0.7
    >>> print(gamma_info)
    GammaInfo(th=array([0., 0.]), M=array([[0.7.+0.j , 0. +0.j ],
            [0. +0.j , 0.7+0.j ]]), s=(0.7+0j), u=array([0.+0.j, 1.+0.j]), 
            v=array([0.-0.j, 1.-0.j]))
    """
    # Perform necessary tests - TODO
    assert isinstance(D, np.ndarray) and isinstance(hD, np.ndarray)

    # perform compression
    D, hD = compress_matrices_delays(D, hD)

    if  hD.size < 2:
        # emtpy delay difference equation or 1 delay -> gamma(r) = 0 by default
        if hD.size == 0:
            M = np.zeros(shape=(0,0))
        else:
            M = np.zeros(shape=(D.shape[0], D.shape[1]))
        gamma_info = GammaInfo(np.zeros(shape=(0,)), M, 0+0j, np.zeros(shape=(0,)), np.zeros(shape=(0,)))
        return 0.0, gamma_info
    
    # number of unique delays in DIFF at least 2
    if hD.size > 3: # -> slow computation warning
        logger.warning(f"Large number of delays in the delay difference equation {hD.size=}. Computing gamma(r) might be slow.")
    
    DD, hDD = normalize_diff(D, hD)

    gamma_val, gamma_info = gamma_normalized_diff(DD, hDD, r, **kwargs)
    return gamma_val, gamma_info
