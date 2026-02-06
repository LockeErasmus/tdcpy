"""
Functions and its gradients for stabilization
---------------------------------------------
TODO:
    1. also implement gradients respecting Ax = b
    2. move gradient test to separate file
    3. implement numerical gradient, as scipy.misc.derivative is depricated in
        upcoming versions
"""

import logging
from typing import Callable
import time
import tdspy as tds

import numpy as np
import numpy.typing as npt
from scipy import linalg, optimize

from tdspy.stability.characteristic_roots import rightmost_root, RightmostRootInfo
from tdspy.stability.spectral_abscissa import spectral_abscissa, spectral_abscissa_diff
from tdspy.common.compress import compress_matrices_delays
from tdspy.stability.gamma_r import gamma_diff, gamma_normalized_diff, func
from tdspy.common.delay_difference_equation import ddae_to_diff, normalize_diff, _normalize_diff

from .utils import diff_dependency_mask

logger = logging.getLogger("__name__")


def func_sa(x: npt.NDArray, E: npt.NDArray, P: npt.NDArray, hP: npt.NDArray, hK: npt.NDArray, Kmask: npt.NDArray, B: npt.NDArray, C: npt.NDArray) -> tuple[float, npt.NDArray]:
    """ function for spectral abscissa and its gradient with respect to
    controller parameters

    The system (Closed Loop) is defined as

        E dxdt(t) = SUM P[i] x(t-hP[i]) + SUM B * K[j] * C x(t-hK[j])       (1)

    where K = x.reshape(Kshape) are controller parameters, hK controller delays,
    P and hP are controlled system matrices and delays, respectively. 

    Args:
        x (array): vectorized controller parameters
        E (array): 2d array defining LHS of Closed Loop
        P (array): 3d array defining the Pi matrices (controlled system)
        hP(array): 1d array defining delays associated with P
        hK(array): 1d array defining delays associated with K (note that
            x := vec(K))
        Kmask(array): 3d array defining the gradient mask for controller
            parameters K, has to have same shape as K
        B(array): left 2d matrix defining the position of controller matrices
            with respect to closed loop, see (1)
        C(array): right 2d matrix defining the position of controller matrices
            with respect to closed loop, see (1)
    
    Returns:
        tuple containing:

            - fval (float): spectral abscissa
            - grad (array): jacobian, 1d array matching shape of x
    """
    # x is 1d array (vectorized K) -> inverse this operations
    K = x.reshape((B.shape[1], C.shape[0], hK.shape[0])) # 3d aray from (1)
    A = np.concatenate( # 3d array containing whole RHS closed loop
        [
            P, # controlled system dynamics + conections
            np.einsum(# more efficient way to obtain B @ K[:,:,i] @ C for all i
                'ijk,jn->ink',
                np.einsum('ni,ijk->njk', B, K),
                C,
            ), # controller dynamics
        ],
        axis=2, # stack by delays axis
    )
    hA = np.r_[hP, hK] # 1d array containing all closed loop delays
    A, hA = compress_matrices_delays(A, hA) # duplicates and unsorted hA
    rmr, rmr_info = rightmost_root(E, A, hA, r=0)

    conj_u_T = np.conj(rmr_info.u[np.newaxis,:]) # u* with shape=(1,n)
    v = rmr_info.v[:,np.newaxis] # v with shape=(n,1)
    dM = E + np.sum(A * hA * np.exp(-rmr*hA), axis=2) # dM(s)/ds evaluated at s=rmr
    den = conj_u_T @ dM @ v # gradient denumenator
    matrix = (conj_u_T @ B).T @ (C @ v).T

    # only real part, see spectral abscissa definition
    fval = np.real(rmr)
    fgrad = np.real(1/den * (Kmask * np.exp(-rmr*hK)) *matrix[:,:,np.newaxis])
    
    # gradient needs to be vectorized to match shape of x
    return fval, fgrad.reshape(-1)

def func_cd_old(x: npt.NDArray, E: npt.NDArray, P: npt.NDArray, hP: npt.NDArray, Kmask: npt.NDArray, hK: npt.NDArray, B: npt.NDArray, C: npt.NDArray, 
            uE: npt.NDArray, vE: npt.NDArray, out: tuple, cd: float) -> tuple[float, npt.NDArray]:
    """ function for strong spectral abscissa of associated delay difference
    equation and its gradient with respect to controller parameters

    The system (Closed Loop) is defined as

        E dxdt(t) = SUM P[i] x(t-hP[i]) + SUM B * K[j] * C x(t-hK[j])       (1)

    where K = x.reshape(Kshape) are controller parameters, hK controller delays,
    P and hP are controlled system matrices and delays, respectively. 

    Args:
        x (array): vectorized controller parameters
        E (array): 2d array defining LHS of Closed Loop
        P (array): 3d array defining the Pi matrices (controlled system)
        hP(array): 1d array defining delays associated with P
        hK(array): 1d array defining delays associated with K (note that
            x := vec(K))
        Kmask(array): 3d array defining the gradient mask for controller
            parameters K, has to have same shape as K
        B(array): left 2d matrix defining the position of controller matrices
            with respect to closed loop, see (1)
        C(array): right 2d matrix defining the position of controller matrices
            with respect to closed loop, see (1)
 
    Returns:
        tuple containing:

            - fval (float): strong spectral abscissa of associated
                delay difference equation
            - grad (array): jacobian, 1d array matching shape of x
    """
    # for DIFF dependency
    from tdspy.stabopt.utils import diff_dependency_mask

    # extract K from x
    K = x.reshape((B.shape[1], C.shape[0], hK.shape[0])) # 3d aray from (1)
    
    # create closed-loop matrices A,hA
    # A = SUM P[i] x(t-hP[i]) + SUM B * K[j] * C x(t-hK[j])
    # hA = [hP, kK]
    
    A = np.concatenate( # 3d array containing whole RHS closed loop
        [
            P, # controlled system dynamics + conections
            np.einsum(# more efficient way to obtain B @ K[:,:,i] @ C for all i
                'ijk,jn->ink',
                np.einsum('ni,ijk->njk', B, K),
                C,
            ), # controller dynamics
        ],
        axis=2, # stack by delays axis
    )
    hA = np.r_[hP, hK] # 1d array containing all closed loop delays
    # A, hA = compress_matrices_delays(A, hA) # duplicates and unsorted hA

    # check for only the DDE here
    
    # here, user specifies adjustability of controller parameters
    Kmask = np.full_like(K, fill_value=True, dtype=bool) # all parameters adjustable

    r = diff_dependency_mask(Kmask, uE, vE, B, C)

    if np.all(~r):
        print("DIFF IS INDEPENDENT OF CONTROLLER PARAMETERS")
        is_dde_dependent = False
    else:
        print("DIFF IS DEPENDENT")
        is_dde_dependent = True

    # create cl_ddae
    cl = tds.DDAE(E=E,A=A,hA=hA)            # remove later
    D, hD = ddae_to_diff(E, A, hA, uE, vE)
    DD, hDD = normalize_diff(D, hD)

    # extract diff and compute cd and gamma0
    diff = cl.get_delay_difference_equation()   

    sa_diff, cdInfo = spectral_abscissa_diff(DD,hDD,r=-0.1)                         # = -0.8657, ok
    gamma0, out = gamma_normalized_diff(DD, hDD, r=0, correction=True, is_compressed=0)

    # extract the zero-delay terms from the dde - ??
    zero_delay = hA==0
    A0 = A[:,:,zero_delay]
    hA0 = hA[zero_delay]
    D0, hD0 = ddae_to_diff(E, A0, hA0, uE, vE)
    DD0, hDD0 = normalize_diff(D0, hD0)

    # first make sure that CD is finite for initial optimization variables
    # if gamma(inf)>1, then CD = inf and the gragInf, info = gamma_normalized_diff(DD0, hDD0, 0, correction=True, n_theta=10)d cannot be computed
    
    

    # if the associated delay-difference equation doesn't depend on the controller parameters
    # compute CD/gamma0
    if is_dde_dependent:
        if D.shape[1] > 1:
            cd, cd_info = spectral_abscissa_diff(DD, hDD, r=-0.1)
            gamma0, gamma0_out = gamma_normalized_diff(DD, hDD, r=0, correction=True, n_theta=10)
        else:
            cd = -np.inf
            gamma0, gamma0_out = 0, []

    else:
        gInf0, ginfo = gamma_diff(A0,hA0,0)
        cd = cd
        if gInf0 >=1:
            raise ValueError("gammaInf > 1, objective function is infeasible for all possible optimization variables")
        else:
            # compute cd
            gammar, gamma_info = gamma_diff(diff.A,diff.hA,r,correction=True,n_theta=10)
            cd, cd_info = spectral_abscissa(diff.A,diff.hA)
        pass

    # obtain parameters [r,l,u',v,th]
    # [gammar,out] = compute_gamma_r(diff.A,diff.hA,r,options)

    # num = (out.uE'*diff)


    raise NotImplementedError(".") # TODO implement

def func_gamma(x: npt.NDArray, E: npt.NDArray, P: npt.NDArray, hP: npt.NDArray, hK: npt.NDArray, Kmask: npt.NDArray, B: npt.NDArray, C: npt.NDArray) -> tuple[float, npt.NDArray]:
    """ function for of gamma(r) and its gradient TODO

    The system (Closed Loop) is defined as

        E dxdt(t) = SUM P[i] x(t-hP[i]) + SUM B * K[j] * C x(t-hK[j])       (1)

    where K = x.reshape(Kshape) are controller parameters, hK controller delays,
    P and hP are controlled system matrices and delays, respectively. 

    Args:
        x (array): vectorized controller parameters
        r TODO
        E (array): 2d array defining LHS of Closed Loop
        P (array): 3d array defining the Pi matrices (controlled system)
        hP(array): 1d array defining delays associated with P
        hK(array): 1d array defining delays associated with K (note that
            x := vec(K))
        Kmask(array): 3d array defining the gradient mask for controller
            parameters K, has to have same shape as K
        B(array): left 2d matrix defining the position of controller matrices
            with respect to closed loop, see (1)
        C(array): right 2d matrix defining the position of controller matrices
            with respect to closed loop, see (1)

    Returns:
        tuple containing:

            - fval (float): TODO
            - grad (array): TODO
    """
    raise NotImplementedError(".") # TODO implement

def func_cd(x: npt.NDArray, E: npt.NDArray, P: npt.NDArray, hP: npt.NDArray,
           Kmask: npt.NDArray, hK: npt.NDArray, B: npt.NDArray, C: npt.NDArray, 
           uE: npt.NDArray, vE: npt.NDArray, **kwargs):
    """ TODO

    Assumptions:
        1. cd is a function of x (and can be changed via changing p)
        2. hP[0] == 0.0, hP are unique and sorted len > 1
    """
    # extract kwargs

    # reshape vector x into 3D array K
    K = x.reshape((B.shape[1], C.shape[0], hK.shape[0])) # 3d aray from (1)

    ## TODO: allow to pass these pre-computed matrices
    KA = np.einsum(# more efficient way to obtain uE.T @ K[:,:,i] @ vE
        'ijk,jn->ink',
        np.einsum('ni,ijk->njk', B, K),
        C,
    )
    A = np.concatenate([P, KA], axis=2)
    hA = np.r_[hP, hK] # 1d array containing all closed loop delays
    D = np.einsum(# more efficient way to obtain uE.T @ K[:,:,i] @ vE
        'ijk,jn->ink',
        np.einsum('ni,ijk->njk', uE.T, A),
        vE,
    )

    # TODO filter out D[:,:,i] close to zero??? TODO
    H, hH = normalize_diff(D, hA)

    # compute cd: TODO filter out H[:,:, i] close to zero?
    cd, cd_info = spectral_abscissa_diff(H, hH)

    # unpack precomputed values from cd function
    theta = cd_info.th # critical values of theta, 0 is prepended -> same length as hH
    u = cd_info.u
    v = cd_info.v
    s = cd_info.s
    # M = cd_info.M

    # evaluate denumerator
    dM = np.sum(
        (
        np.einsum( # evalueate SUM uH @ Di @ v 
                'ijk,jn->ink',
                np.einsum('ni,ijk->njk', np.conj(u)[np.newaxis,:], H),
                v[:, np.newaxis],
            )
            * hH * np.exp(-cd * hH) * np.exp(1j*theta)
        )
    )
    den = np.real(np.conj(s) * dM / np.inner(np.conj(u), v))

    # evaluate u* D_0^{-1} U^T B
    lu, piv = linalg.lu_factor(D[:,:,0])
    left_matrix = np.conj(u)[np.newaxis,:] @ linalg.lu_solve((lu, piv), uE.T @ B)
    
    # evaluate (C V v)^T
    right_matrix = C @ vE @ v[:, np.newaxis]

    ix = len(hP)
    matrix = left_matrix.T @ right_matrix.T
    vector = np.conj(s) * np.exp(-cd*hH[ix-1:]) * np.exp(1j * theta[ix-1:]) # shape(mH,)
    array = matrix[:,:, np.newaxis] * vector[np.newaxis, :]
    # array[:,:,k] = matrix * vector[k]\ 

    dK = (1./den) * np.real(array)

    # mask gradients
    dKmasked = np.where(Kmask, dK, 0)
    return cd, dKmasked


def grad_gamma0(x: npt.NDArray, DP: npt.NDArray, hDP: npt.NDArray, 
                Kmask: npt.NDArray, hK: npt.NDArray, 
                BU: npt.NDArray, CV: npt.NDArray) -> tuple[float, npt.NDArray]:
    """ gradient of gamma0 function for delay-difference equations

    Computes the gradient of the function :math:`\gamma_0` of the delay-difference
    equation defined by matrices D and delays hD with respect to controller
    parameters.
    For the normalized DDE defined as
    .. math::
        0 = I x(t) + H_1(p) x(t-h_1) + ... + H_m(p) x(t-h_m),

        with H_i = linalg.solve(D[:, :, 0], D[:, :, i]), hH_i = hD[i].

    the gradient is given by

    .. math::
    
        \nabla_{p_k} \gamma_0(p) = \frac{1}{|\lambda|} \text{Re} \left( \bar{\lambda} u^* \left( \frac{\partial H_1(p)}{\partial p_k} +
                         \sum_{i=2}^m \frac{\partial H_i}{\partial p_k} e^{j\theta_i}  \right) w \right).

    Parameters
    ----------
    x : npt.NDArray
        optimization variables
    DP : npt.NDArray
        difference equation matrices 
    hDP : npt.NDArray
        difference equation delays
    Kmask : npt.NDArray
        mask for the controller entries
    hK : npt.NDArray
        controller delays
    BU : npt.NDArray
        left matrix defining position of controller in DDE (= U.T @ B)
    CV : npt.NDArray
        right matrix defining position of controller in DDE (= C @ V)

    Returns
    -------
    A tuple containing:

        - fval : float 
                spectral abscissa
        - grad : array
                gradient, 1d array matching shape of x

    Notes
    -----
    The function computes the gradient of the function :math:`\gamma_0` of the delay-difference equation defined by matrices D and delays hD with respect to controller
    parameters.
    For the normalized DDE defined as
    .. math::
        0 = I x(t) + H_1 x(t-h_1) + ... + H_m x(t-h_m),

        with H_i = linalg.solve(D[:, :, 0], D[:, :, i]), hH_i = hD[i].

    Examples
    --------
    

    """
    from tdspy.stabopt.utils import diff_dependency_mask
    from tdspy.common.delay_difference_equation import normalize_diff
    from tdspy.stability.gamma_r import gamma_normalized_diff, gamma_diff
    from scipy import linalg, optimize

    # unpack arguments
    n, nh, nu, ny = DP.shape[0], DP.shape[2], BU.shape[1], CV.shape[0]
    
    # set controller parameters
    K = x.reshape((nu, ny, hK.shape[0]))
    
    # form D = D_P + B_D @ K_D @ C_D
    BK  = np.einsum('lm,mki->lki', BU, K)   # B @ K_i for all i
    BKC = np.einsum('lki,kn->lni', BK, CV)  # (B @ K_i) @ C

    D, hD = np.concatenate([DP, BKC], axis=2), np.concatenate([hDP, hK], axis=0)
    DD, hDD = normalize_diff(D, hD)

    # compute gamma0 and associated info
    g0, gInfo = gamma_normalized_diff(DD, hDD, r=0, correction=True, n_theta=10)
    s, u, v, th = gInfo.s, gInfo.u, gInfo.v, gInfo.th

    # initialize gradient
    grad = np.zeros_like(x)

    ###################### Gradient computation ######################

    if s == 0:
        logger.warning("WARNING: s == 0, gradient is ill-defined")
        return g0, grad
    

    # The DDE is of the form
    # 0 = I x(t) + DD[:,:,0] x(t-h1) + DD[:,:,2] x(t-h2) + ... + DD[:,:,m] x(t-hm)
    # u* dH/dp v = (u^* @ linalg.solve(D[:,:,0], U.T @ B) ).T @ ((C @ V) @ v).T * np.sum(exp(j*th[0])),

    # uDBU = (u^* @ linalg.solve(D[:,:,0], U.T @ B) ).T
    uDBU = (np.conj(u).T[np.newaxis,:] @ linalg.solve(D[:,:,0], BU)) # dimensions (nu,)

    # # CVv = ((C @ V) @ v).T
    CVv = (CV @ v[:,np.newaxis]) # dimensions (ny,)

    # # u * dH/dp v = uDBU * CVv * SUM(exp(j*th))
    # udH_dpv = (uDBU * CVv)  # dimensions (nu, ny)
    # grad = np.real(np.conj(s) * uDBU.T * CVv.T * np.sum(np.exp(1j*th[m-1:]))) / np.abs(s)
    
    # # the below divides by |u^* v| (not in the original formula, but needed for correct scaling)
    # grad = grad / np.real(np.conj(u).T @ v) 

    m = len(hDP)

    matrix = uDBU.T @ CVv.T
    vector = np.conj(s) * np.exp( 1j * th[m-1:] ) # shape (mH,)
    array = matrix[:,:, np.newaxis] * vector[np.newaxis, :]

    grad = (1 / np.abs(s)) * np.real(array) / np.real(np.conj(u).T @ v)

    # mask gradients
    grad_masked = np.where(Kmask, grad, 0)


    return g0, grad_masked.reshape(-1)



def gradient_test(func: Callable, x: npt.NDArray, args: tuple, h: float=1e-4, tolerance: float=1e-6):
    """ function to test the numerical accuracy of the computed gradient using central differences
    The method uses central differences for testing the numerical gradient.
    
    Args: TODO
        func:   cost function
        h:      step size
        E, P, hP, hK, Kmask, B, C:  arguments


    Returns:

        tuple containing:
            - fgrad : npt.NDArray
                analytical gradient
            - fgrad_num : npt.NDArray
                numerical gradient

    Notes
    -----
    The function computes the numerical gradient using central differences and compares it to the analytical gradient provided by the function ``func``.
    If the norm of the difference between the two gradients is less than the specified tolerance, the test is considered passed.   

    Examples
    --------
    >>> import numpy as np
    >>> from tdspy.stabopt.gradients import gradient_test
    >>> def func_example(x, E, P, hP, hK, Kmask, B, C):
    ...     # example function returning cost and gradient
    ...     cost = np.sum(x**2)  # simple quadratic cost
    ...     grad = 2*x           # gradient of the cost
    ...     return cost, grad
    >>> x0 = np.array([1.0, 2.0, 3.0])
    >>> E = np.eye(2)
    >>> P = np.random.rand(2, 2, 2)
    >>> hP = np.array([0.1, 0.2])
    >>> hK = np.array([0.1, 0.2])
    >>> Kmask = np.ones((2, 2, 2), dtype=bool)
    >>> B = np.random.rand(2, 2)
    >>> C = np.random.rand(2, 2)
    >>> fgrad, fgrad_num = gradient_test(func_example, x0, (E, P, hP, hK, Kmask, B, C))
    Gradient test passed norm=0.0 < 1e-06

    """

    # analytical gradient
    s = time.perf_counter()
    _, fgrad = func(x, *args)
    perf_analytical = time.perf_counter() - s
    
    # numerical gradient
    s = time.perf_counter()
    fgrad_num =  np.zeros_like(x) # numerical gradient
    for i in range(0, x.shape[0]):
        x_forward = np.copy(x)
        x_backward = np.copy(x)
        x_forward[i] += h
        x_backward[i] -= h

        f_forward, _ = func(x_forward, *args)
        f_backward, _ = func(x_backward, *args)

        fgrad_num[i] = (f_forward - f_backward)/(2*h)
    perf_numerical = time.perf_counter() - s
    

    diff = np.linalg.norm(fgrad - fgrad_num)
    if diff < tolerance:
        print(f"Gradient test passed norm={diff} < {tolerance}")
        print(f"Gradient computation peformance:\n    ANALYTICAL:     {perf_analytical} [s]\n     NUMERICAL:     {perf_numerical} [s]")
    else:
        print(f"Gradient test failed. Difference: {diff}")

    return fgrad, fgrad_num


