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
from tdspy.common.delay_difference_equation import ddae_to_diff, normalize_diff

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
    fgrad = np.real(1/den * (Kmask * np.exp(-rmr*hK)) * matrix[:,:,np.newaxis])
    
    # gradient needs to be vectorized to match shape of x
    return fval, fgrad.reshape(-1)

def func_cd(x: npt.NDArray, E: npt.NDArray, P: npt.NDArray, hP: npt.NDArray, Kmask: npt.NDArray, hK: npt.NDArray, B: npt.NDArray, C: npt.NDArray, 
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

def gradient_test(func: Callable, x: npt.NDArray, args: tuple, h: float=1e-4, tolerance: float=1e-6):
    """ function to test the numerical accuracy of the computed gradient using central differences
    The method uses central differences for testing the numerical gradient.
    
    Args: TODO
        func:   cost function
        h:      step size
        E, P, hP, hK, Kmask, B, C:  arguments
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
