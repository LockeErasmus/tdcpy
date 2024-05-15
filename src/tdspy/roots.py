"""
Implementation of tds_roots
"""
import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg

from .rdde import RDDE
from .ddae import DDAE
from .ndde import NDDE
from .stability.discretization_heuristic import compute_n_rhp, compute_n_rect
from .stability.bounds import lower_bound, upper_bound
from .stability.newton import newton_correction
from .common.discretization import discretize
from .gamma_r import gamma_r

logger = logging.getLogger(__name__)

def roots(tds: DDAE, r=0.0, **kwargs):
    """

    Args:
        tds (TODO)
        r (int or list): region, specify r as number on real axis or rectangular
            region via 4 coordinates [Re_min, Re_max, Im_min, Im_max], default r=0.0

    kwargs:
        max_size_evp (int): TODO, default 600
        discretization (int): discretization, if None heuristic is envoked,
            default None, keep default if you don't know, has to be > 1
        basic_delay (float): define if delays are commensurate, default None,
            used in discretization heuristic case `rhp`
    """

    # checks for region definition
    if isinstance(r, (int, float)):
        # all OK
        case = "rhp"
    elif isinstance (r, list):
        assert len(r) == 4, "region has to be defined in form [a,b,c,d]"
        assert r[0] < r[1] and r[2] < r[3], "region has to be defined as [a,b,c,d], a<b, c<d"
        # TODO assert all from r finite ?
        case = "rect"
    else:
        raise ValueError(("Region (argument `r`) has to be defined as number, "
                          "example `r=-5.1` or rectangular region, example "
                          "`[-5, 10.5, 0, 100]`."))
    
    # type of TDS, RDDE and DDAE -> OK, NDDE -> convert to DDAE
    if isinstance(tds, NDDE):
        tds = tds.to_ddae()

	# compress tds (this also sorts)
    tds = tds.compress()

    # unpack TDS object
    n = tds.n
    E = tds.E

    if tds.mA == 0:
        hA = np.array([0.0], dtype=E.dtype)
        A = np.zeros(shape=(n,n,1), dtype=E.dtype)
    elif tds.hA[0] > 0:
        hA = np.r_[0, hA] # prepend 0.0 delay
        A = np.concatenate([np.zeros(shape=(n,n,1), dtype=E.dtype), A], axis=2)
    else:
        hA = tds.hA
        A = tds.A

    # CASE 1: ODE or DAE (no delays)
    if hA.shape[0] == 1:
        logger.debug("CASE: ODE or DAE -> finite number of roots")
        # just use eig and filter based on rhp or region
        result = linalg.eig(A[:,:,0], E, left=False, right=False)
        if case == "rhp":
            mask = np.isfinite(result) & (np.real(result)>=r)
            return result[mask]
        else: # case == "rect"
            mask = (np.isfinite(result) & (np.real(result)>=r[0]) 
                    & (np.real(result)<=r[1]) & (np.imag(result)>=r[2])
                    & (np.imag(result)<=r[3]))
            return result[mask]

    # CASE 2:
    mA = hA.shape[0] # number of delay terms

    # scale tds such that maximal delay-value equals 1
    # lambda_hat = lambda*tau_m
    # det(lambda E - A0 - A1 *exp(-lambda tau_1) - ... - Am *exp(-lambda tau_m)) = 0
    # => det(lambda_hat E - tau_m *A0 - tau_m * A1 *exp(-lambda_hat tau_1/tau_m) - ... - tau_m Am *exp(-lambda_hat)) = 0
	# re-scaled system matrices and delays are stored in K and tau_s
    tau_max = hA[-1] # last delays is maximal one
    tau_s = hA / tau_max # scale tau vector
    K = tau_max * A # scale matrices of dynamics

    ###########################################################
    # Heuristic for N (degree of the spectral discretisation) #
    ###########################################################    
    # obtain discretization
    discretization = kwargs.get("discretizaton", None)
    max_size_evp = kwargs.get("max_size_evp", 600)
    max_size_evp_enforced = False # flag to indicate that size of EVP > max_size_evp
    assert n <= max_size_evp, "The size of the delay differential equation exceeds max_size_evp"
    discretization_max = int(np.floor(max_size_evp / n) - 1)
    
    if case == "rhp":
        # rescale r
        rs = r * tau_max
        # introduce shift of the origin, shifted matrices B, C
        B = K[:,:,0] + (-rs)*E
        C = K[:,:,1:] * np.exp(-rs * tau_s[1:])
        if discretization is None: # envoke heuristic
            if False: # TODO line 289 - 295, as of now unimportant, later KWARG
                # condition C_D > r is assumed to be already checked
                ...
            else:
                diff = tds.to_delay_difference_equation()
                if diff is not None: # empty associated delay difference equation (E is not singular)
                    if diff.hA[0] != 0 or False: # TODO
                        raise ValueError("The provided DDAE does not satisfy assumption 2.1.")

                    if gamma_r(diff, r) >= 1.0:
                        discretization = 30
                        logger.warning((f"Gamma_r exceeds {gamma_r} >= 1 (i.e., CD>r). Spectral "
                                        "discretization with N = 30 (lowered if maximum size of "
                                        "eigenvalue problem is exceeded). Try specifying a "
                                        "rectangular region instead."))
                
                if discretization is None:
                    # discretization is still undefined, reason:
                    #   (a) - no underlying delay-difference equation or
                    #   (b) - gamma_r < 1.0
                    # => region RHP contains finitely many roots and heuristic
                    #    can be applied
                    basic_delay = kwargs.get("basic_delay", None)
                    discretization = compute_n_rhp(E, B, C, tau=hA, basic_delay=basic_delay)
            
            # check if discretization does exceed limit
            if discretization > discretization_max:
                max_size_evp_enforced = True
                discretization = discretization_max
                size_evp = n*(discretization_max + 1)
                logger.warning(
                    (f"Size of the generalized EVP would exceed its maximum "
                     f"value. Discretization around {max([0,r])} + 0*1j with N "
                     f"= {discretization} instead (size of new eigenvalue "
                     f"problem: {size_evp} x {size_evp}. As a consequence, not "
                     "all characteristic roots in the specified right "
                     "half-plane might be found. To make sure that all desired "
                     "characteristic roots are found, either increase r (i.e., "
                     "shift the desired right half-plane to the right) or "
                     "increase the kwargs 'max_size_evp'. For more information "
                     "consult the documentation of this function.")
                )

        else: # discretization provided by user -> peform checks
            assert isinstance(discretization, int), "discretization has to be int"
            assert discretization > 1, "discretization has to be > 1"
            logger.debug(f"User provided {discretization=}")
        
        if max_size_evp_enforced and r < 0:
            rs = 0
            QQ=K
        else: # use shift rs
            QQ = np.concatenate([B[:,:,np.newaxis], C], axis=2)

    else: # case == "region":
        if discretization is None: # envoke rectangular region heuristic
            discretization, origin = compute_n_rect(r, tau_max)
            
            # check if discretization does exceed limit
            if discretization > discretization_max:
                max_size_evp_enforced = True
                discretization = discretization_max
                size_evp = n*(discretization_max + 1)
                logger.warning(
                    (f"Size of the generalized EVP would exceed its maximum "
                     f"value. Discretization around {np.real(origin)/tau_max} + "
                     f"{np.imag(origin)/tau_max}j with N = {discretization} "
                     f"instead (size of new eigenvalue problem: {size_evp} x "
                     f"{size_evp}. As a consequence, not all characteristic "
                     "roots in the specified right half-plane might be found. "
                     "To make sure that all desired characteristic roots are "
                     "found, either increase r (i.e., shift the desired right "
                     "half-plane to the right) or increase the kwargs "
                     "'max_size_evp'. For more information consult the "
                     "documentation of this function.")
                )
        else:
            assert isinstance(discretization, int), "discretization has to be int"
            assert discretization > 1, "discretization has to be > 1"
            origin = tau_max * ((r[0]+r[1])/2) + 1j*((r[2]+r[3])/2)
            logger.debug(f"User provided {discretization=} | {origin=} ")
        
        QQ = np.copy(A)
        QQ[:,:,0] =  K[:,:,0] + (-origin)*E
        QQ[:,:,1:] = K[:,:,1:] * np.exp(-origin * tau_s[1:])

    logger.info(f"Degree of spectral discretization is N = {discretization}")
    
    ########################### 
	# Spectral discretisation #
	###########################
    # [3] Jarlebring, E., Meerbergen, K., & Michiels, W. (2010). A Krylov
    #     method for the delay eigenvalue problem. SIAM Journal on Scientific
    #     Computing, 32(6), pp. 3278-3300.  Section 2.2.

    # create DDAE and discretize into DAE
    ddae = DDAE(E=E, A=QQ, hA=tau_s)
    dae = discretize(ddae, discretization)

    # solve EVP
    raw_roots = linalg.eig(dae.A, dae.E, left=False, right=False)
    raw_roots = raw_roots[np.isfinite(raw_roots)] # get rid of inf and NaN

    # undo shift and scaling
    if case == "rhp":
        raw_roots += rs
    else: # case == "rect"
        raw_roots += origin
    raw_roots = raw_roots / tau_max

    ######################
	# Newton corrections #
	######################
    # TODO - line 441

    # Select characteristic roots for Newton corrections
    if case == "rhp":
        mask = ((np.real(raw_roots) >= lower_bound(r, 0.1, 0.1)) 
                & (np.imag(raw_roots) >= 0.0)) # due to symetry, drop imag < 0
        newton_roots0 = raw_roots[mask]
    else: # case == "rect"
        mask = ((np.real(raw_roots)>=lower_bound(r[0], 0.1, 0.1))
                & (np.real(raw_roots)<=upper_bound(r[1], 0.1, 0.1))
                & (np.imag(raw_roots)>=lower_bound(r[2], 0.1, 0.1))
                & (np.imag(raw_roots)<=upper_bound(r[3], 0.1, 0.1)))
        newton_roots0 = raw_roots[mask]
    
    newton_roots = newton_correction(newton_roots0, E, A, hA, inplace=False)


    # # TODO solve if newton roots emtpy
    # newton_roots0 = np.copy(newton_roots)


    # newton_max_iterations=2
    # newton_abs_tol = 1e-8
    # for i in range(newton_roots.shape[0]):
    #     newton_roots[i] # initial guess eigen value
    #     #print(newton_roots[i])
    #     R_lambda = (newton_roots[i] * E - A[:,:,0]
    #                 - np.sum(A[:,:,1:]*np.exp(-newton_roots[i]*hA[1:]), axis=2))
    #     (U, s, Vh) = linalg.svd(R_lambda, compute_uv=True)
    #     #print(f"Matrix M")
    #     #print(str(R_lambda))
    #     #print("")
    #     #print(s)
    #     #print(linalg.eig(R_lambda))
    #     v0 = Vh[-1]
    #     #print(v0)
        

    #     residual = linalg.norm(R_lambda @ v0[:, np.newaxis], ord=None, axis=None) # 2-norm of np.ravel(.) is returned
    #     print(f"iteration ---, {residual=}, {newton_roots[i]}")
    #     if residual <= newton_abs_tol:
    #         pass # TODO already sufficiently close
    #     else:
    #         # netwton is envoked to increase precission
    #         jacobian = np.zeros(shape=(n+1, n+1), dtype=R_lambda.dtype)
    #         jacobian[-1, :n] = np.conj(v0)
    #         jacobian[-1, -1] = 0
            
    #         dR_lambda = np.zeros_like(R_lambda)
    #         v = np.copy(v0) #
    #         f_val = np.zeros(shape=(n+1,), dtype=R_lambda.dtype)

    #         for j in range(newton_max_iterations):

    #             dR_lambda = E + np.sum(A[:,:,1:]*(hA[1:]*np.exp(-newton_roots[i]*hA[1:])), axis=2)
    #             # update jacobian
    #             jacobian[:n, :n] = R_lambda
    #             jacobian[:n, -1] = (dR_lambda @ v[:, np.newaxis])[:,0] # TODO check if this is efficient

    #             f_val[:n] = (-R_lambda @ v[:, np.newaxis])[:,0]
    #             f_val[-1] = -(np.inner(v0, v) - 1) # v0 is already stored in jacovian[-1, :n]

    #             (dx, _, _, _) = linalg.lstsq(jacobian, f_val[:,np.newaxis])
    #             dx = linalg.inv(jacobian)@f_val[:,np.newaxis]

    #             # update - TODO
    #             newton_roots[i] += dx[-1, 0]
    #             v += dx[:n, 0]

    #             # check converged?
    #             residual = linalg.norm(R_lambda @ v[:, np.newaxis], ord=None, axis=None)
    #             print(f"iteration {j=}, {residual=}, {newton_roots[i]}")
    #             if residual <= newton_abs_tol:
    #                 break # converged

    #             # update R_lambda
    #             R_lambda = (newton_roots[i] * E - A[:,:,0]
    #                 - np.sum(A[:,:,1:]*np.exp(-newton_roots[i]*hA[1:]), axis=2))
            

        



    # # continue 476



    return newton_roots, newton_roots0

    








    
    


