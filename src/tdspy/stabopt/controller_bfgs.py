"""
Delay controller design (stabilization via BFGS)
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg, optimize

from tdspy.stability.characteristic_roots import rightmost_root, RightmostRootInfo
from tdspy.common.compress import compress_matrices_delays
from .gradients import func_sa, func_cd, func_gamma
from tdspy.controller import create_static_controller, interconnect3
from tdspy.common.composition import concatenate_2x2_by_delays
from tdspy import DDAE, ClosedLoop  
from tdspy.stability.gamma_r import gamma_diff, gamma_normalized_diff, func
from tdspy.stability.spectral_abscissa import spectral_abscissa_diff

logger = logging.getLogger("__name__")

def design_bfgs(E: npt.NDArray, P:npt.NDArray, hP:npt.NDArray, K0, hK, B, C, **kwargs):
    """
    Args:
        TODO
        **kwargs:
            mask (array): masking gradient
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

def stab_opt(ddae,nc,**kwargs):
    """
    function to optimize the closed-loop spectral abscissa of:
        ddae
    returns the controller gain matrix K, CL
    Args: 
        ddae:   open-loop system
        nc:     degree of the dynamic controller

    kwargs: 
        basis:          controller structure
        mask (array):   gradient mask
        options:        stabilization options
    """

    # unpack arguments
    n, nu, ny = ddae.n, ddae.n_inputs, ddae.n_outputs

    # get default settings
    initial = kwargs.get("initial", np.zeros(shape=(nc+nu,nc+ny,1),dtype=float))        # shape is not valid if there are controller delays!!!
    hK = np.zeros([0.])
    y_indices = kwargs.get("y_indices",np.arange(0,ny-1))                               # y_indices for closed-loop
    u_indices = kwargs.get("u_indices",np.arange(0,nu-1))                               # u_indices for closed-loop
    Kmask = kwargs.get("mask", np.full_like(initial, fill_value=True, dtype=bool))      # create mask
    nc = kwargs.get("nc",0)                                                             # nc
    options = kwargs.get("options",{"nstart",1,"r",-1,"Ntheta",10,"w1",0.001,"w2",0.001,"fvalquit",-np.inf})   # get_options

    # stability options: nstart, Ntheta, fvalquit, w1, w2
    # options = kwargs.get("options",{})  

    # verify existing controller mask, ny, nu, nc
    assert initial.shape[0]==nu+nc, "dimension of input channels must match controller dimension"
    assert initial.shape[1]==ny+nc, "dimension of output channels must match controller dimension"

    # verify mask - to be programmed later

    # create closed loop, ddae + controller
    cl = ClosedLoop(ddae, nc, y_indices, u_indices, K0=initial, hK=hK)
    E = cl.E
    P = cl._A
    hP = cl._hA
    K0 = K0
    hK0 = cl.hK
    B = cl.BB
    C = cl.CC

    # extract cl_ddae and diff
    cl_ddae = DDAE(E=cl.E,A=cl.A,hA=cl.hA)

    # check if cl is_retarded
    if cl_ddae.is_essentially_retarded:
        # tds is retarded, minimize the spectral abscissa function c
        # however, this does not take into account infinitesimal delay perturbations
        func = func_sa
        pass

    else:
        # tds is neutral, check dependency
        # check if delay diff equation depends on K
        from tdspy.stabopt.utils import diff_dependency_mask

        diff = cl_ddae.get_delay_difference_equation()
        r = diff_dependency_mask(Kmask, cl.uE, cl.vE, cl.BB, cl.CC)
        
        # case 1: dde 
        if diff.A.shape[2] == 1:  # system is retarded
            func = func_sa          
        else:                       # system is neutral
            func = func_cd          


        if all(~r):
            dependency_flag = 0     # diff is independent of controller parameters

            # compute cd, gamma0 just once
            cd = cd(cl_ddae)
            gamma0 = gamma_normalized_diff(diff.A[:,:,1:], diff.hA[1:], 0, correction=True)

            if cd>0:    # system is not stabilizable, the system is unstable
                print(f"cd={cd} and independent of controller parameters. System cannot be stabilized!")
                pass
            else:
                sol = design_bfgs(E, P, hP, K0, hK, B, C, options={"disp": True, "eps":0.1})
               
        else:
            dependency_flag = 1     # diff is dependent on controller parameters

            # check for feasibility
            # extract the zero-delay terms from the dde - ??

            feasibility_flag = check_feasibility(diff)

             # first make sure that CD is finite for initial optimization variables
            # if gamma(inf)>1, then CD = inf and the grad cannot be computed
            gInf, info = gamma_diff(diff0.A[:,:,1:], diff0.hA[1:], 0, correction=True, n_theta=10)
            gInf0, ginfo = gamma_diff(A0,hA0,0)
            cd = cd
            if gInf0 >=1:
                raise ValueError("gammaInf > 1, objective function is infeasible for all possible optimization variables")
            else:
                # compute cd
                gammar, gamma_info = gamma_diff(diff.A,diff.hA,r,correction=True,n_theta=10)
                cd, cd_info = spectral_abscissa_diff(diff.A,diff.hA)
                
        

        pass

    def check_feasibility(diff):
        """
        function to check feasibility of the optimization variables
        sub-function of stab_opt, returns a boolean
        Difference Equation:
        DD[:,:,0] + DD[:,:,1] x(t-hDD[1]) + ... + DD[:,:,]
        Args:
            diff: delay-difference equation

        """
        zero_delay = diff.hA==0
        A0 = diff.A[:,:,zero_delay]
        hA0 = diff.A[:,:,zero_delay]
        diff0 = ddae(diff.E,A0,hA0) 

        # first make sure that CD is finite for initial optimization variables
        # if gamma(inf)>1, then CD = inf and the grad cannot be computed
        gInf, info = gamma_diff(diff0.A[:,:,1:], diff0.hA[1:], 0, correction=True, n_theta=10)
        gInf0, ginfo = gamma_diff(A0,hA0,0)
        cd = cd
        if gInf0 >=1:
            feasibility_flag = False
            # raise ValueError("gammaInf > 1, objective function is infeasible for all possible optimization variables")
        else:
            # compute cd
            feasibility_flag = True
            gammar, gamma_info = gamma_diff(diff.A,diff.hA,r,correction=True,n_theta=10)
            cd, cd_info = spectral_abscissa_diff(diff.A,diff.hA)

    