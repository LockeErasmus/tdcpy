# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha
# Copyright (C) 2026 Diego Garcia

r"""
GRANSO: GRadient-based Algorithm for Non-Smooth Optimization
------------------------------------------------------------

Python implementation of GRANSO algorithm for constrained nonsmooth
optimization problems. It was developed by Curtis, Mitchel and Overton (see
http://www.timmitchell.com/software/GRANSO/) specifically for the problems
like our eigen-value problem (nonconvex, non-locally Lipschitz). For more
details, see the following paper:

Curtis, F.E., Mitchell, T. and Overton, M.L., 2017. A BFGS-SQP method for
nonsmooth, nonconvex, constrained optimization and its evaluation using
relative minimization profiles. Optimization Methods and Software, 32(1),
pp.148-181.


Also note, there is already Python implementation of GRANSO:

    - https://github.com/sun-umn/PyGRANSO
    - https://ncvx.org/

But it is based on pytorch and implemented in a very "heavy" style, making its
usage inconvenient for us.
"""

import numpy as np
import numpy.typing as npt
from typing import Callable
import traceback

try:
    import osqp
except ImportError as e:
    raise ImportError(
        "The 'osqp' package is required for using GRANSO but not installed.\n"
        "Install it with:\n\n"
        "    pip install osqp\n\n"
        "or, if you use conda:\n\n"
        "    conda install -c conda-forge osqp"
    ) from e


def _line_search_weak_wolfe(func: Callable, x0, f0, g0, p, c1: float=0., c2: float=0.5, alpha0: float=1.0, maxit: int|None=None,
                            eval_limit: int|None=None, step_tol: float=1e-12, fval_quit: float|None=None):
    """

    func returns:

        fval, fgrad, is_feasible
    
    x0: x at 0
    f0: f(x0) - function value at x0
    g0: gradient evaluated at x0
    p: direction of search

    alpha0: initial guess for step-length

    maxit: int default 100, None --> runs for ever
    eval_limit: maximum allowed functions evals --> None runs for ever
    """
    # TODO consider normalizing direction at the beggining?

    # step-length settings
    alpha = alpha0 # initial step-length
    alpha_lb, alpha_ub = 0.0, np.inf # bounds for step-length
    
    xalpha = np.copy(x0)
    falpha = np.copy(f0)
    galpha = np.copy(g0)

    g0_dot_p = np.dot(np.conj(g0), p) 

    p_norm = np.linalg.norm(p)

    n_evals = 0
    n_iters = 0
    n_expand = 0

    expand_limit: int = max(10, round(np.log2(1e5/p_norm))) # this allows more expansions if norm of direction is small

    while (maxit is None or n_iters < maxit) and \
          (eval_limit is None or n_evals < eval_limit):
        # I WILL ADD CODE HERE TODO
        # n_evals += 1 when ever function is called TODO -> decorator?
        if (alpha_ub - alpha_lb) > np.linalg.norm(x0 + alpha_lb * p) / p_norm * step_tol:
            break
        
        xalpha = x0 + alpha * p

        falpha, galpha, is_feasible = func(xalpha)
        n_evals += 1
        galpha_dot_p = np.dot(np.conj(galpha), p)

        if (
            fval_quit is not None
            and is_feasible
            and np.isfinite(falpha)
            and falpha <= fval_quit
        ):
            status = 0
            return (alpha, xalpha, falpha, galpha, status)
        
        # Check Wolfe conditions
        # C1: Armijo condition (sufficient decrease) with ">="
        if np.isnan(falpha) or falpha >= f0 + c1 * alpha * g0_dot_p:
            alpha_ub = alpha # step-length too large, adjust alpha UB
        # C2: Curvature condition 
        elif np.isnan(galpha_dot_p) or galpha_dot_p < c2 * g0_dot_p:
            alpha_lb = alpha
        else: # Both conditions satisfied
            alpha_lb = alpha
            alpha_ub = alpha
            status = 0
            return (alpha, xalpha, falpha, galpha, status)
        
        # Set up new func eval
        if not np.isinf(alpha_ub):
            alpha = 0.5 * (alpha_lb + alpha_ub) # apply bisection
        elif n_expand < expand_limit: # alpha_ub == np.inf -> expanding
            n_expand += 1
            alpha *= 2
        else: # expansion limit reached
            break

    # Line search failed
    if np.isinf(alpha_ub): # minimizer never bracketed
        status = 2
    else: # bracketed, but never satisifed both Wolfe conditions
        status = 1

    return (alpha, xalpha, falpha, galpha, status)

def _solve_steering_dual_qp(H, f, lb, ub, mu_Hinv_f_grad, Hinv_c_grads) -> npt.NDArray:
    try:
        options = {
        "eps_abs": 1e-12,
        "eps_rel": 1e-12,
        "polish": True,
        "verbose": False,
    }
        prob = osqp.OSQP()
        prob.setup(H, f, None, None, lb, ub, **options)
        res = prob.solve()
        y = res.x
    except Exception as e:
        print("QP solver failed solving steering dual QP: Steering aborted")
        traceback.format_exc()
    
    d = -mu_Hinv_f_grad - (Hinv_c_grads @ y)
    return d

def _predicted_violation_reduction(d: npt.NDArray, violation, ci: npt.NDArray, ci_grad: npt.NDArray,
                                   ce: npt.NDArray, ce_grad: npt.NDArray, norm_ord=1):
    dL = (violation
          - np.linalg.norm(np.clip(ci + ci_grad.T @ d, 0, None), ord=norm_ord)
          - np.linalg.norm(ce + ce_grad @ d, ord=norm_ord))
    return dL

def _sqp_steering_strategy(mu, f_grad, ci, ci_grad, ce, ce_grad, apply_Hinv: Callable, 
                           violation: float, norm_ord: int=1, c_viol=0.01, c_mu: float=0.5,
                           maxiter: int=40, ineq_margin: float=0) -> tuple[npt.NDArray, float]:
    # Steps:
    #   1. compute feasibility step once
    #   2. steering loop
    #   3. fallback

    n_ineq = len(ci)  
    n_eq = len(ce)

    # TODO: obtain Hinv_f_grad as = Hinv @ f_grad but as function handle???
    # TODO: better will be to just pass bfgs object?
    Hinv_f_grad = apply_Hinv(f_grad)

    violation_tol = np.sqrt(np.finfo(np.float64).eps) * max(violation, 1)

    # prepare for QP
    c_grads = np.hstack((ce_grad, ci_grad))
    Hinv_c_grads = apply_Hinv(c_grads)
    H = np.conj(c_grads.T) @ Hinv_c_grads
    H += np.conj(H.T) # perform Hermitian projection (memory efficient)
    H *= 0.5
    mu_Hinv_f_grad = mu * Hinv_f_grad
    f = np.conj(c_grads.T) @ mu_Hinv_f_grad - np.vstack( (ce, ci) )
    lb = np.vstack( (-np.ones(shape=(n_eq, 1)), np.zeros(shape=(n_ineq, 1))) )
    ub = np.ones(shape=(n_eq + n_ineq, 1))

    # Optimality step
    d = _solve_steering_dual_qp(H, f, lb, ub, mu_Hinv_f_grad, Hinv_c_grads)
    reduction = _predicted_violation_reduction(d, violation, ci, ci_grad, ce, ce_grad, norm_ord=norm_ord)

    if reduction >= c_viol * violation - violation_tol:
            return d, mu, reduction
    
    # If no equality constraints AND all inequality constraints feasible, we
    # disable steering loop.
    if n_eq == 0 and ineq_margin != np.inf and np.all(ci < -ineq_margin):
        return d, mu, reduction
    
    #  Predicted violation reduction was inadequate. Check to see
    #  if reduction is an adequate fraction of the predicted reduction 
    #  when using the reference direction (given by the QP with the 
    #  objective removed, that is, with the penalty parameter temporarily 
    #  set to zero)

    # Feasibility step
    mu_Hinv_f_grad = np.zeros_like(Hinv_f_grad)
    f = - np.vstack( (ce, ci) )
    d_ref = _solve_steering_dual_qp(H, f, lb, ub, mu_Hinv_f_grad, Hinv_c_grads)
    reduction_ref = _predicted_violation_reduction(d_ref, violation, ci, ci_grad, ce, ce_grad, norm_ord=norm_ord)
    
    if reduction_ref <= violation_tol:
        # No predicted feasibility improvement possible.
        # Accept optimality step.
        return d, mu, reduction

    if reduction >= c_viol * reduction_ref - violation_tol:
        return d, mu, reduction
    
    # Steering loop:
    #  - lower mu (penalty parameter) to produce new search direction
    #  - check if predicted violation reduction is sufficient
    for _ in range(maxiter):
        mu *= c_mu
        mu_Hinv_f_grad = mu * Hinv_f_grad
        f = np.conj(c_grads.T) @ mu_Hinv_f_grad - np.vstack( (ce, ci) )
        d = _solve_steering_dual_qp(H, f, lb, ub, mu_Hinv_f_grad, Hinv_c_grads)
        reduction = _predicted_violation_reduction(d, violation, ci, ci_grad, ce, ce_grad, norm_ord=norm_ord)
        if reduction >= c_viol * reduction_ref - violation_tol:
            return d, mu, reduction

    # Fallback:
    #  - all directions failed
    #  - as per original GRANSO implementation, use last one
    #  - last direction corresponds to the lowest mu (penalty parameter)
    # TODO: consider warning here?
    return d, mu, reduction

from scipy.optimize import OptimizeResult

def granso(fun, x0, args=(), jac=None, hess=None, constraints=(), bounds=None,
           callback=None, options=None) -> OptimizeResult:
    """
    Based on:

    Frank E. Curtis, Tim Mitchell, and Michael L. Overton,
     A BFGS-SQP method for nonsmooth, nonconvex, constrained
     optimization and its evaluation using relative minimization
     profiles, Optimization Methods and Software, 32(1):148-181, 2017.
     Available at https://dx.doi.org/10.1080/10556788.2016.1208749

    """
    
    raise NotImplementedError("Not implemented yet")
    return OptimizeResult(...)