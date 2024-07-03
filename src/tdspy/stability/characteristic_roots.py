"""
Characteristic roots
--------------------
Set of functionalities connected to characteristic roots of DDAE

Implemented functions:
    1. roots_ddae -> characteristic roots of DDAE
    1. rightmost_root -> right most root and necessary things for gradient

"""

from collections import namedtuple
import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg

from tdspy.common.delay_difference_equation import ddae_to_diff, normalize_diff
from tdspy.common.discretization import discretize_ddae
from .bounds import lower_bound, upper_bound
from .gamma_r import gamma_normalized_diff, gamma_diff
from .discretization_heuristic import compute_n_rhp, compute_n_rect
from .newton import newton_correction

logger = logging.getLogger(__name__)

RootsInfo = namedtuple("RootsInfo", ["discretization", "gamma_r_exceeds_one", "index_exceeds_one",
                                     "discretization_eigenvalues", "max_size_evp_enforced",
                                     "newton_inital_guesses", "newton_final_values", 
                                     "newton_residuals", "newton_unconverged_initial_guesses", 
                                     "newton_large_corrections"])

RightmostRootInfo = namedtuple("RightmostRootInfo", ["M", "DM", "u", "v", "found", "max_size_evp_enforced"])

def roots_ddae(E: npt.NDArray, A: npt.NDArray, hA: npt.NDArray, r: float,  **kwargs):
    
    # TODO asserts

    # checks for region definition
    if isinstance(r, (int, float)):
        # all OK
        case = "rhp"
    elif isinstance (r, list):
        assert len(r) == 4, "region has to be defined in form [a,b,c,d]"
        assert r[0] < r[1] and r[2] < r[3], "region has to be defined as [a,b,c,d], a<b, c<d"
        assert np.all(~np.isinf(r)), "region has to be finite rectangle"
        case = "rect"
    else:
        raise ValueError(("Region (argument `r`) has to be defined as number, "
                          "example `r=-5.1` or rectangular region, example "
                          "`[-5, 10.5, 0, 100]`."))

    n = E.shape[0]
    if hA[0] > 0:
        hA = np.r_[0, hA] # prepend 0.0 delay
        A = np.concatenate([np.zeros(shape=(n,n,1), dtype=E.dtype), A], axis=2)

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
    discretization = kwargs.get("discretization", None)
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
                D, hD = ddae_to_diff(E, A, hA)
                print(D, hD)
                if hD.size != 0: # delay difference equation exists (E is singular)
                    # DD, hDD = normalize_diff(D, hD)
                    if hD[0] != 0 or False: # TODO
                        raise ValueError("The provided DDAE does not satisfy assumption 2.1.")

                    gamma_val, gamma_info = gamma_diff(D, hD, r)
                    if gamma_val >= 1.0:
                        discretization = 30
                        logger.warning((f"gamma(r; ...)= {gamma_val} exceeds 1 (i.e., CD>r). Spectral "
                                        "discretization with N = 30 (lowered if maximum size of "
                                        "eigenvalue problem is exceeded). Try specifying a "
                                        "rectangular region instead."))
                
                if discretization is None:
                    # discretization is still undefined, reason:
                    #   (a) - no underlying delay-difference equation or
                    #   (b) - gamma(r) < 1.0
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
        
        QQ = np.full_like(A, fill_value=0, dtype=np.complex128)
        QQ[:,:,0] =  K[:,:,0] - origin*E
        QQ[:,:,1:] = K[:,:,1:] * np.exp(-origin * tau_s[1:])

    logger.info(f"Degree of spectral discretization is N = {discretization}")
    
    ########################### 
	# Spectral discretisation #
	###########################
    # [3] Jarlebring, E., Meerbergen, K., & Michiels, W. (2010). A Krylov
    #     method for the delay eigenvalue problem. SIAM Journal on Scientific
    #     Computing, 32(6), pp. 3278-3300.  Section 2.2.

    # create DDAE and discretize into DAE
    Pi_N, Sigma_N = discretize_ddae(E, QQ, tau_s, discretization) # E, A
    
    # solve EVP
    raw_roots = linalg.eig(Sigma_N, Pi_N, left=False, right=False)
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
    # Select characteristic roots for Newton corrections
    # TODO, sometimes this drop can cause drop double roots with Im part close to 0-
    if case == "rhp":
        mask = ((np.real(raw_roots) >= lower_bound(r, 0.1, 0.1)) 
                & (np.imag(raw_roots) >= -1e-10)) # due to symetry, drop imag < 0 TODO kwarg
        newton_roots0 = raw_roots[mask]
    else: # case == "rect"
        mask = ((np.real(raw_roots) >= lower_bound(r[0], 0.1, 0.1))
                & (np.real(raw_roots) <= upper_bound(r[1], 0.1, 0.1))
                & (np.imag(raw_roots) >= lower_bound(r[2], 0.1, 0.1))
                & (np.imag(raw_roots) <= upper_bound(r[3], 0.1, 0.1)))
        newton_roots0 = raw_roots[mask]
    
    # perform newton corrections
    newton_roots, residuals, converged_mask, correction_large_mask = newton_correction(newton_roots0, E, A, hA)
    newton_roots = newton_roots[np.isfinite(newton_roots)] # get rid of inf and NaN

    if case == "rhp":
        # add back conjugates, but not those close to real 0 axis
        mask0 = ~np.isclose(np.imag(newton_roots), 0, rtol=0, atol=1e-10)
        roots = np.r_[newton_roots, np.conj(newton_roots[mask0])]
        roots = roots[roots >= r]
        if roots.size == 0:
            logger.warning(f"No characteristic roots found in right half-plane (Re(z)>= {r})")
    else: # case == "rect"
        mask = ((np.real(newton_roots) >= r[0])
                & (np.real(newton_roots) <= r[1])
                & (np.imag(newton_roots) >= r[2])
                & (np.imag(newton_roots) <= r[3]))
        roots = newton_roots[mask]
        if roots.size == 0:
            logger.warning(f"No characteristic roots found in rectangular region {r}")

    # prepare RootsInfo TODO - fix gamma_r and index bools
    info = RootsInfo(discretization, False, False, raw_roots, max_size_evp_enforced, newton_roots0, newton_roots, residuals, converged_mask, correction_large_mask)

    return roots, info

def rightmost_root(E: npt.NDArray, A: npt.NDArray, hA: npt.NDArray, r: float,  **kwargs):
    """ Computes the rightmost root of compressed DDAE represented via E, A, hA
    in {z \in C: Re(z) >= r && Im(z) >= 0}.

    Args:
        E (array): TODO
        A (array): TODO
        hA (array): TODO
        r (float): TODO
        **kwargs: TODO
    
    Returns:
        rightmost_root (complex)
        --- do not forget to also return M, dM, u, v -> gradient computation
    """

    # TODO unpack kwargs

    # TODO assert
    # TODO assert all real values in E, A, hA

    # find all roots via discretization
    cr, cr_info = roots_ddae(E, A, hA, r)

    # find right most root
    root_star = None
    root_star_found = False # indicator, that right most root succesfully found

    # CASE 1: roots cr are empty
    if cr.size == 0:
        # try to use eigenvalues of discretized DDAE
        u = np.sort_complex(cr_info.discretization_eigenvalues) # in ascending order

        # check whether the approximations obtained using spectraldiscretisation
        # make the characteristic matrix sufficiently close to being singular.
        for r in u[::-1]: # iterate from r with largest Re(r)
            # evaluate characteristic matrix
            M = r*E + np.sum(A * np.exp(-hA*r), axis=2)

            # perform SVD and check what is the smallest eigen value
            U, s, Vh = linalg.svd(M, compute_uv=True) # sorted in non-increasing order
            v0 = np.conj(Vh[-1]) # eigen vector associated to smallest eigenvalue
            residual = linalg.norm(M @ v0[:, np.newaxis], ord=None, axis=None) # 2-norm of np.ravel(.) is returned

            if residual <= 1e-6: # i.e. eigenvalue is sufficiently close to 0.0, TODO kwargs
                root_star = s[-1]
                root_star_found = True
                break
    
        if root_star is None: # no root to the right of r found
            u_smaller_r = u[np.real(u) < r]
            if u_smaller_r.size == 0: # worst case scenario, rutrn fallback value
                root_star = r + 0j
            else: # take the rightmost root to the left of r
                root_star = u_smaller_r[-1] # already sorted, take last element

    # CASE 2: the rightmost point in cr lies significantly to the left of
    # the rightmost point eigen value from discretization
    elif np.max(np.real(cr)) <= lower_bound(np.max(np.real(cr_info.discretization_eigenvalues)), 0.05, 1.e-3):
        cr_discretization = cr_info.discretization_eigenvalues
        imax = np.argmax(np.real(cr_discretization))
        root_star1 = cr_discretization[imax]

        # check the roots possibly right of root_star1
        u = np.vectorize(upper_bound)(np.real(cr_discretization), 0.05, 1.e-3)
        u = np.sort_complex(u)

        # check whether the approximations obtained using spectraldiscretisation
        # make the characteristic matrix sufficiently close to being singular.
        for r in u[::-1]: # iterate from r with largest Re(r)
            # evaluate characteristic matrix
            M = r*E + np.sum(A * np.exp(-hA*r), axis=2)

            # perform SVD and check what is the smallest eigen value
            U, s, Vh = linalg.svd(M, compute_uv=True) # sorted in non-increasing order
            v0 = np.conj(Vh[-1]) # eigen vector associated to smallest eigenvalue
            residual = linalg.norm(M @ v0[:, np.newaxis], ord=None, axis=None) # 2-norm of np.ravel(.) is returned

            if residual <= 1e-6: # i.e. eigenvalue is sufficiently close to 0.0, TODO kwargs
                root_star = s[-1]
                root_star_found = True
                break
        
        if root_star is None: # default to rightmost
            root_star = root_star1

    # CASE 3: newton corrections for rightmost root converged inside `roots_ddae(.)`
    else: 
        i = np.argmax(np.real(cr))
        root_star = cr[i]
        root_star_found = True
    
    # evaluate M(root_star), M'(root_star), u(root_star) and v(root_star)
    M = root_star * E - np.sum(A * np.exp(-root_star*hA), axis=2)
    DM = E + np.sum(A * hA * np.exp(-root_star*hA), axis=2)
    U, _, Vh = linalg.svd(M, compute_uv=True)
    u = U[:,-1]
    v = np.conj(Vh[-1])
    
    root_info = RightmostRootInfo(M, DM, u, v, root_star_found, cr_info.max_size_evp_enforced)

    return root_star, root_info