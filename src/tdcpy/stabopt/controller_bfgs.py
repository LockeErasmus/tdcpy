# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

"""
Delay controller design (stabilization via BFGS)
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg, optimize

from tdcpy.common.delay_difference_equation import ddae_to_diff
from tdcpy.stability.characteristic_roots import rightmost_root, RightmostRootInfo
from tdcpy.common.compress import compress_matrices_delays
from .gradients import func_sa, func_cd, grad_gamma0
from tdcpy.controller import create_static_controller, interconnect3
from tdcpy.common.composition import concatenate_2x2_by_delays
from tdcpy import DDAE, ClosedLoop  
from tdcpy.stability.gamma_r import gamma_diff, gamma_normalized_diff, func
from tdcpy.stability.spectral_abscissa import spectral_abscissa_diff

logger = logging.getLogger("__name__")

def design_bfgs(E: npt.NDArray, P:npt.NDArray, hP:npt.NDArray, K0, hK, B, C, **kwargs):
    """
    function for designing a controller via BFGS optimization

    Parameters
    ----------
    E :      array
        E matrix of closed-loop system
    P :      array
        system matrix of open-loop system
    hP :     array
        system delays of open-loop system
    K0 :     array
        initial controller gain matrix
    hK :     array   
        controller delays
    B :      array
        input matrix of closed-loop system
    C :      array
        output matrix of closed-loop system
    kwargs : dict
        options:    options for the optimization solver

    Returns
    -------
    sol : OptimizeResult
        Optimization result containing the optimal controller parameters and optimization information
        x: optimal controller parameters
        fun: optimal objective function value
        success: whether the optimization was successful
        message: description of the cause of the termination
    
    Notes
    -----
    1. This function assumes that the system is retarded.
    2. The optimization problem is non-convex, and the solution may depend on the initial controller parameters K0.
    3. The optimization is performed using the BFGS algorithm, which is a quasi-Newton method for unconstrained optimization. The objective function is the spectral abscissa of the closed-loop system, which is computed using the function `func_sa`. The gradient of the objective function is computed using the function `grad_sa`.
    4. The optimization options can be specified via the `options` key in `kwargs`. 

    Examples
    --------
    >>> from tdcpy.stabopt.controller_bfgs import design_bfgs
    >>> import numpy as np
    >>> E = np.eye(2)
    >>> P0 = np.array([[-1., 0.], [0., -2.]])
    >>> P1 = np.array([[-0.5, 0.], [0., -0.5]])
    >>> P = np.stack([P0, P1], axis=2)
    >>> hP = np.array([0.,0.5])
    >>> K0 = np.zeros((2,2,1))
    >>> hK = np.array([0.])
    >>> B = np.eye(2)
    >>> C = np.eye(2)
    >>> sol = design_bfgs(E, P, hP, K0, hK, B, C, options={"disp": True})

    """

    Kmask = kwargs.get("mask", np.full_like(K0, fill_value=True, dtype=bool))
    sol = optimize.minimize(
        func_sa,
        K0.reshape(-1),
        args=(E, P, hP, hK, Kmask, B, C),
        jac=True,
        method=kwargs.get("method", "L-BFGS-B"),
        options=kwargs.get("options", {}),
        callback=kwargs.get("callback", None)
    )
    return sol

def design_granso(E: npt.NDArray, P:npt.NDArray, hP:npt.NDArray, K0, hK, B, C, **kwargs):
    """
    Args:
        TODO
        **kwargs:
            mask (array): masking gradient
    """

    import traceback
    import time
    import torch
    from torch import linalg as LA
    from pygranso.pygranso import pygranso as granso
    import scipy.io
    from pygranso.pygransoStruct import pygransoStruct

    device = torch.device('cpu')
    double_precision = True
    torch_dtype = torch.double
    print_level = 0

    # variables and corresponding dimensions
    # torch.zeros(p*m,1).to(device=device, dtype=torch.double)
    n = K0.reshape(-1).shape[0]
    var_in = {"x": [n,1]}
    Kmask = kwargs.get("mask", np.full_like(K0, fill_value=True, dtype=bool))

    # define user_options
    opts = pygransoStruct()
    # opts.x0 = torch.tensor(K0.reshape(-1), dtype=torch_dtype, device=device)
    # opts.x0 = torch.zeros_like(K0.reshape(-1).to(torch_dtype), dtype=torch_dtype, device=device)
    opts.x0 = 4*torch.ones(n,1, dtype=torch_dtype, device=device)    
    opts.torch_device = device
    opts.print_frequency = 10
    opts.maxit = 200
    opts.globalAD = False

    # define obj. function
    def obj_fn(X_struct,E,P,hP,hK,Kmask,B,C):
        # decision variables
        X = X_struct.x.detach().numpy()
        n = X.shape[0]

        # objective function
        f, g = func_sa(X, E, P, hP, hK, Kmask, B, C)
        
        grad = torch.from_numpy(g.reshape([n,1]))

        # inequality constraint
        ci = None
        ci_grad = None

        # equality constraint
        ce = None
        ce_grad = None

        return [f, grad, ci, ci_grad, ce, ce_grad]
        
    comb_fn = lambda X_struct : obj_fn(X_struct,E, P, hP, hK, Kmask, B, C)

    start = time.time()
    sol = granso(var_spec = var_in, combined_fn=comb_fn, user_opts=opts)
    end = time.time()

    print("Total Time: {}s".format(end-start))

    return sol

def minimize_spectral_abscissa(ddae: DDAE, order: int, **kwargs):
    """
    function to controller parameters for minimizing the spectral abscissa of a ddae
    
    Parameters
    ----------
    ddae : DDAE
        DDAE object representing the open-loop system 
    order : int
        Degree of the dynamic controller
    kwargs : dict
        initial (array):    initial controller parameters, shape (nc+nu, nc+ny, nd)
        y_indices (array):  indices of system outputs used for feedback
        u_indices (array):  indices of system inputs used for control
        basis :             controller structure
        mask (array):       gradient mask
        options:            stabilization options

    Returns
    -------
    sol : OptimizeResult
        Optimization result containing the optimal controller parameters and optimization information
        x: optimal controller parameters
        fun: optimal objective function value
        success: whether the optimization was successful
        message: description of the cause of the termination

    Notes
    -----
    1. The optimization problem is non-convex, and the solution may depend on the initial controller parameters.
    2. The default solver if L-BFGS-B. 
    3. The optimization options can be specified via the `options` key in `kwargs`.
    

    Examples
    --------
    
    >>> from tdcpy.stabopt.controller_bfgs import minimize_spectral_abscissa
    >>> from tdcpy.ddae import DDAE
    >>> A0 = np.array([[-1., 0.], [0., -2.]])
    >>> A1 = np.array([[1., 0.], [0., 1.]])
    >>> A = np.stack([A0, A1], axis=2)
    >>> hA = np.array([0., 1.])
    >>> Bu = np.array([ [-0.1],[-0.2]    ])
    >>> B = np.stack([Bu],axis=2)
    >>> hB = np.array([0.5])
    >>> C = np.array(np.eye(2))
    >>> C = np.stack([C],axis=2)
    >>> hC = np.array([0])
    >>> D = np.zeros(shape=(2,1,1), dtype=float)    # must be defined, otherwise error!
    >>> hD = np.array([0.])
    >>> ddae = DDAE(A=A,hA=hA,B=B,hB=hB,C=C,hC=hC,D=D,hD=hD)
    >>> sol = minimize_spectral_abscissa(ddae, order=0, method="L-BFGS-B", options={"disp": True}, type = "barrier")

    """

    # import necessary functions
    from tdcpy.ddae import DDAE
    from tdcpy.stabopt.controller_bfgs import design_bfgs

    ##################################### Step 0: Preprocessing and unpacking arguments ######################################
    
    # unpack arguments
    n, nu, ny = ddae.n, ddae.n_inputs, ddae.n_outputs

    # get default settings
    n_delays = kwargs.get("n_delays", 1)                                                     # number of controller delays, default is 1 (zero-delay)
    K0 = kwargs.get("K0", np.zeros(shape=(order+nu,order+ny,n_delays),dtype=float))         # initial controller parameters, default is all zeros
    hK = kwargs.get("hK", np.zeros(shape=(n_delays,), dtype=float))                         # controller delays, default is no delay
    assert hK.shape[0] == n_delays, "hK must have length n_delays!"                     # check hK shape 

    Kmask = kwargs.get("mask", np.full_like(K0, fill_value=True, dtype=bool))               # get mask for controller parameters, default is all True
    assert Kmask.shape == K0.shape, "Mask shape must match K0 shape!"                       

    callback = kwargs.get("callback", None)                                                     # callback function for optimization, default is None
    type = kwargs.get("type", "barrier")                                                              # optimization type, default is barrier
    method = kwargs.get("method", "L-BFGS-B")                                                     # optimization method, default is L-BFGS-B
    options = kwargs.get("options", {"disp": True, "eps":0.1, "gtol": 1e-6, "ftol": 1e-12, "maxls": 100})   # optimization options, default is some reasonable settings for BFGS optimization

    # verify existing controller mask, ny, nu, nc
    assert K0.shape[0]==nu+order, "dimension of input channels must match controller dimension"
    assert K0.shape[1]==ny+order, "dimension of output channels must match controller dimension"
    
    if K0.ndim < 3 or K0.shape[2] != n_delays:  # Check if the third dimension is missing or not equal to n_delays  
        if K0.ndim < 3:
            K0 = K0[:, :, np.newaxis]   # Add the third dimension if it doesn't exist
        if K0.shape[2] != n_delays:
            # Adjust the third dimension to match n_delays
            K0 = np.resize(K0, (K0.shape[0], K0.shape[1], n_delays))

    y_indices = kwargs.get("y_indices",np.arange(0,ny))                               # y_indices for closed-loop
    u_indices = kwargs.get("u_indices",np.arange(0,nu))                               # u_indices for closed-loop
    nc = kwargs.get("order",0)                                                             # nc
    options = kwargs.get("options",{"nstart":1,"Ntheta":10,"w1":0.1,"w2":0.001,"fvalquit":-np.inf})   # get_options
    # Ntheta: 
    # w1: acceptable region for gamma0, i.e., gamma0 < 1 - w1, default is 0.001
    # w2: weight for log-barrier term in the objective function, default is 0.001, meaning that we want to balance between minimizing spectral abscissa and ensuring feasibility (gamma0 < 1 - w1)
    # fvalquit: objective function value to quit optimization, default is -inf, meaning no early stopping based on objective function value
    # gn: target gamma for barrier optimization, default is 0.5

    optmization_options = kwargs.get("options",{"disp": True, "eps":0.1, "gtol": 1e-6, "ftol": 0, "maxls": 100})   # get_options
    

    #################################### Step 1: Initialization ######################################
    
    # if all initial parameters are zero, set initial Ac = 0.1*I, Dc = 0.1*I, Bc = 0, Cc = 1 as a default initial controller structure 
    # this is to ensure that the initial controller is not identically zero, which can cause issues in the optimization (e.g., zero gradient, infeasibility)
    
    if np.all(K0==0): 

        K0[:order,:order,:] = 0.1*np.ones(shape=(order,order,n_delays))       # initial Ac = 0.1*I
        K0[:order,order:order+ny,:] = np.zeros(shape=(order,ny,n_delays))     # initial Bc
        K0[order:order+nu,:order,:] = np.ones(shape=(nu,order,n_delays))      # initial Cc
        K0[order:order+nu,order:order+ny,:] = 0.01*np.ones(shape=(nu,ny,n_delays)) # initial Dc

    # form closed loop, ddae + controller
    cl = ClosedLoop(ddae, order, y_indices, u_indices, K0=K0, hK=hK)
    E = cl.E
    uE = cl.uE
    vE = cl.vE
    P = cl._A
    hP = cl._hA
    K0 = K0
    hK0 = cl.hK
    B = cl.BB
    C = cl.CC

    # extract cl_ddae and diff
    cl_ddae = DDAE(E=cl.E,A=cl.A,hA=cl.hA)

    ################################### Step 2: Pre-checks ##################################
    
    # check if cl is_retarded
    if cl_ddae.is_essentially_retarded:

        ############################### Case 1: Retarded system ###############################
        
        # tds is retarded, minimize the spectral abscissa function c
        # however, this does not take into account infinitesimal delay perturbations
        
        ################################# Optimization #####################################

        sol = design_bfgs(E, P, hP, K0, hK, B, C, method=method, options={"disp": True}, callback=None)
        return sol

    else:

        ################################## Case 2: Neutral system ###############################
        
        # import necessary functions
        from tdcpy.stability.gamma_r import gamma_diff, gamma_normalized_diff, func
        from tdcpy.stability.spectral_abscissa import spectral_abscissa_diff
        from tdcpy.stabopt.utils import diff_dependency_mask
        from tdcpy.common.delay_difference_equation import normalize_diff

        ################################# Step 2.1: Check feasibility and gradient existence #####################################

        # precomputations
        BU = uE.T @ B
        CV = C @ vE
        P = cl._A
        hP = cl._hA
        x0 = K0.reshape(-1)
        
        # check if delay diff equation depends on K
        diff = cl_ddae.get_delay_difference_equation()  # extract delay difference equation of closed-loop system
        DD, hDD = normalize_diff(diff.A, diff.hA)       # normalize the delay difference equation, extract DD and hDD
        r = diff_dependency_mask(Kmask, uE, vE, B, C)     # get mask for controller parameters affecting the delay difference equation
        
        if np.all(~r): 
            
            ############################# Case 1: No dependency on controller parameters #############################
            
            # compute cd for initial controller parameters, if cd > 0, system is not stabilizable, 
            # if cd <= 0, system is stabilizable but no optimization can be done since diff is independent of controller parameters
            
            cd,_ = spectral_abscissa_diff(DD, hDD)

            if cd>0:    # system is not stabilizable, the system is unstable
                print(f"cd={cd} and independent of controller parameters. System cannot be stabilized!")
                pass
            else:       # delay-difference equation is stable, optimization conducted on spectral abscissa only 
                sol = design_bfgs(E, P, hP, K0, hK, B, C, options={"disp": True, "eps":0.1})
                return sol

        else: # diff is dependent on controller parameters

            ############################# Case 2: Dependency on controller parameters #############################

            # feasibility check - is gamma0 < 1?

            gamma0,_ = gamma_normalized_diff(DD,hDD,0,correction=True,n_theta=10)

            if gamma0 >=1:
                # current controller parameters are infeasible
                print(f"gamma0={gamma0} >= 1 at initial controller parameters. Finding a feasible point!")
                # find a feasible point
                feasible_sol = find_feasible_point(E, P, hP, K0, hK, B, C, options={"disp": True, "eps":0.1}, nstart=5, gamma0_threshold=0.5)

                if feasible_sol is None:
                    print("Could not find a feasible point, trying minimize_CD instead.")
                    return None 
                

            ############################### Optimization ######################################


            if type == "barrier":
                ################ f_objective = alpha - w2*log(1 - w1 - gamma0(p))##################
                ################ gradient = grad_alpha + w2*1/(1 - w1 - gamma0(p)) * grad_gamma0 ##################

                w2 = options.get("w2", 0.001)
                w1 = options.get("w1", 0.001)

                gamma0_args = (E, P, hP, Kmask, hK, B, C)
                sa_args = (E, P, hP, hK, Kmask, B, C)

                log_points = np.logspace(0,-8,9)   # different values of w2 to try, default is [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
                log_points = 6.5e-3 * (0.3 ** np.arange(10))   # 6.5e-3, 1.95e-3, 5.85e-4, ...
                best = None
                eps = 1e-8
                results = []

                def obj_fn(x, gamma0_args, sa_args, options):

                    w1 = options.get("w1", 0.001)
                    w2 = options.get("w2", 0.001)
                    g0, grad_g0 = grad_gamma0(x, *gamma0_args)
                    sa, grad_sa = func_sa(x, *sa_args)         
                    
                    f1 = sa
                    grad_f1 = grad_sa
                    
                    slack = 1 - w1 - g0
                    
                    if slack <= 1e-12 or (not np.isfinite(slack)):
                        return np.inf, np.zeros_like(x)
                        
                    
                    f2 = np.log(slack+eps)
                    grad_f2 = -grad_g0 / slack
                    
                    f = f1 - w2*f2
                    grad = grad_sa - w2*grad_f2

                    if (not np.isfinite(f)) or (not np.all(np.isfinite(grad))):
                        return np.inf, np.zeros_like(x)



                    return (f, grad) 

                x=x0

                for w2 in log_points:
                    print(f"Optimizing with w2={w2}...")
                    sol = optimize.minimize(
                        obj_fn,
                        x0=x,
                        args=(gamma0_args, sa_args,{"w1": w1, "w2": w2}),
                        jac=True,
                        method=method,
                        options=options,
                        callback=None
                    )

                    # logger.info(f"Optimization with w2={w2} completed. Optimal fval={sol.fun}, optimal gamma0={grad_gamma0(sol.x, *gamma0_args)}, optimal sa={func_sa(sol.x, *sa_args)[0]}")
                    results.append((w2, sol.fun, sol.x))
                    x = sol.x  # warm start the next optimization with the current solution

                return sol
            
            elif type == "CD":

                cd_args = (E, P, hP, Kmask, hK, B, C, uE, vE)
                sa_args = (E, P, hP, hK, Kmask, B, C)

                def obj_fn(x, cd_args, options):

                    cd, grad_cd = func_cd(x, *cd_args) 
                    sa, grad_sa = func_sa(x, *sa_args)    
                    
                    f = np.max([cd, sa])
                    
                    if cd>sa:
                        grad = grad_cd
                    else:
                        grad = grad_sa

                    return (f, grad) 
                
                sol = optimize.minimize(
                    obj_fn,
                    x0=x0,
                    args=(cd_args, options),
                    jac=True,
                    method=method,
                    options=options,
                    callback=None
                )

                return sol

                       
    return sol

    # def check_feasibility(diff):
    #     """
    #     function to check feasibility of the optimization variables
    #     sub-function of stab_opt, returns a boolean
    #     Difference Equation:
    #     DD[:,:,0] + DD[:,:,1] x(t-hDD[1]) + ... + DD[:,:,]
    #     Args:
    #         diff: delay-difference equation

    #     """
    #     zero_delay = diff.hA==0
    #     A0 = diff.A[:,:,zero_delay]
    #     hA0 = diff.A[:,:,zero_delay]
    #     diff0 = ddae(diff.E,A0,hA0) 

    #     # first make sure that CD is finite for initial optimization variables
    #     # if gamma(inf)>1, then CD = inf and the grad cannot be computed
    #     gInf, info = gamma_diff(diff0.A[:,:,1:], diff0.hA[1:], 0, correction=True, n_theta=10)
    #     gInf0, ginfo = gamma_diff(A0,hA0,0)
    #     cd = cd
    #     if gInf0 >=1:
    #         feasibility_flag = False
    #         # raise ValueError("gammaInf > 1, objective function is infeasible for all possible optimization variables")
    #     else:
    #         # compute cd
    #         feasibility_flag = True
    #         gammar, gamma_info = gamma_diff(diff.A,diff.hA,r,correction=True,n_theta=10)
    #         cd, cd_info = spectral_abscissa_diff(diff.A,diff.hA)

def find_feasible_point(E: npt.NDArray, P:npt.NDArray, hP:npt.NDArray, K0, hK, B, C, **kwargs):
    """
    function to find a feasible point for the optimization problem

    .. :math::

        minimize    gamma_0(K)
        subject to  gamma_inf(K) < 1

    for a DDAE of the form:
    
    .. :math::
        E \dot{x}(t) = (P_0 + B K_0 C) x(t) + \sum_{i=1}^{m} (P_i + B K_i C) x(t - \tau_i)

    Parameters
    ----------
    E :     array
        E matrix of closed-loop system
    P :     array
        system matrix of open-loop system
    hP :    array
        system delays of open-loop system
    K0 :    array
        initial controller gain matrix
    hK :    array
        controller delays
    B :     array
        input matrix of closed-loop system
    C :     array
        output matrix of closed-loop system
    kwargs :
        options:    options for the optimization solver
        gamma0_threshold: threshold for gamma0 to consider a point feasible, default is 1.0

    Returns
    -------
    K : array
        feasible controller parameters, if found, otherwise None

    Notes
    -----

    
    Examples
    --------
    >>> import numpy as np
    >>> from tdcpy.stabopt.controller_bfgs import find_feasible_point
    >>> E = np.array([[1,0,0],[0,1,0],[0,0,0]])
    >>> P = np.zeros((3,3,1))
    >>> hP = np.array([0])
    >>> K0 = np.zeros((1,3,1))
    >>> hK = np.array([0])
    >>> B = np.eye(3)[:,:,np.newaxis]
    >>> C = np.eye(3)[:,:,np.newaxis]
    >>> K_feasible = find_feasible_point(E, P, hP, K0, hK, B, C)
    >>> print(K_feasible)

    """
    
    from scipy import linalg, optimize
    from tdcpy.common.delay_difference_equation import ddae_to_diff
    from tdcpy.stability.gamma_r import gamma_diff, gamma_normalized_diff
    from tdcpy.common.delay_difference_equation import normalize_diff
    from tdcpy.stabopt.utils import diff_dependency_mask
    from tdcpy.stabopt.gradients import grad_gamma0

    # unpack arguments
    n, nu, ny = E.shape[0], B.shape[1], C.shape[0]

    # get mask
    Kmask = kwargs.get("mask", np.full_like(K0, fill_value=True, dtype=bool))
    options = kwargs.get("options",{"disp": True})   # get_options
    gamma_threshold = kwargs.get("gamma0_threshold", 1.0)
    nstart = kwargs.get("nstart", 1)

    # extract uE and vE from E
    uE = linalg.null_space(E.T,rcond=1e-12)
    vE = linalg.null_space(E,rcond=1e-12)

    ########################## Step 1: Precomputation ##########################

    # extract delay-difference equation without controller contribution
    DP, hDP = ddae_to_diff(E,P,hP)

    # extract controller parameters affecting the delay-difference equation
    r = diff_dependency_mask(Kmask, uE=uE, vE=vE, B=B, C=C)

    # precompute BD, CD
    BD = uE.T @ B
    CD = C @ vE
    x0 = K0.reshape(-1)


    ########################## Step 2: Preliminary checks ##########################
    # obtain A, hA from P+K, hP+hK
    BKC = np.einsum('lm,mki,kn -> lni', B, K0, C)
    A, hA = np.concatenate([P, BKC], axis=2), np.concatenate([hP, hK], axis=0)

    # Retarded or neutral system check
    D, hD = ddae_to_diff(E,A,hA)
    if hD.shape[0] == 1 and hD[0] == 0.0:    # system is retarded
        print("The closed-loop system is retarded, no need to find a feasible point.")
        return x0

    # Feasibility check
    DD,hDD = normalize_diff(D,hD)
    g0, gInfo = gamma_normalized_diff(DD, hDD, r=0, correction=True, n_theta=10)
    # gamma0, info = gamma_diff(D[:,:,1:], hD[1:], 0, correction=True, n_theta=10)
    
    if g0 < 1:
        print("Initial controller parameters are already feasible.")
        return x0
    
    # Dependency check
    if not np.any(r):
        raise ValueError("The delay difference equation is independent of the controller parameters, no feasible point exists.")    
    
    # # control parameters affecting the delay difference equation
    # KD = np.zeros_like(Kmask, dtype=float)
    # np.place(KD, indices, x0)

    ############################# Step 3: Optimization ##########################

    for i in range(nstart):

        print(f"Finding a feasible point, start {i+1}/{nstart}...")

        sol = optimize.minimize(
            grad_gamma0,
            x0,
            args=(E, P, hP, Kmask, hK, B, C),
            jac=True,
            method=kwargs.get("method", "L-BFGS-B"),
            options=options,
            callback=kwargs.get("callback", None)
        )

        if sol.fun < gamma_threshold:
            break

        x0 = np.random.uniform(low=-1., high=1., size=x0.shape)  # random restart

    if sol.fun >= gamma_threshold:
        print(f"Optimization did not find a feasible point, gamma0={sol.fun} >= {gamma_threshold} at optimal controller parameters.")
        return None
    
    return sol
