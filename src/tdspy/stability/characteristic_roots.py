"""
Characteristic roots
--------------------
Set of functionalities connected to characteristic roots of DDAE

Implemented functions:
1. roots_ddae: characteristic roots of DDAE
2. rightmost_root: right most root and necessary things for gradient

TODO:
1. split rootd_ddae into roots_ddae_rhp and roots_ddae_region and move 
higher logic into high level API (tdspy.roots)
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
from tdspy.common.compress import compress_matrices_delays

logger = logging.getLogger(__name__)

RootsInfo = namedtuple("RootsInfo", ["discretization", "gamma_r_exceeds_one", "index_exceeds_one",
                                     "discretization_eigenvalues", "max_size_evp_enforced",
                                     "newton_inital_guesses", "newton_final_values", 
                                     "newton_residuals", "newton_unconverged_initial_guesses", 
                                     "newton_large_corrections"])

RightmostRootInfo = namedtuple("RightmostRootInfo", ["M", "DM", "u", "v", "found", "max_size_evp_enforced"])

def roots_ddae(E: npt.NDArray, A: npt.NDArray, hA: npt.NDArray, r: float | list,  **kwargs):
    """ Computes the roots of DDAE represented via E, A, hA in region r

    TDS is assumed to be defined via E, A, hA as

    .. math::

        E \Dot{x}(t) = \sum\limits_{k=1}^N x(t-h_{A,k})  \ldots \qquad (1)
    
    region is defined either by (1) r is float:
        region = {z \in C: Re(z) >= r && Im(z) >= 0}
    or r is list of 4 floats  [Re_min, Re_max, Im_min, Im_max]
        region = {z \in C: Re(z) \in [Re_min, Re_max] and
                           Im(z) \in [Im_min, Im_max]}

    Parameters
    ----------
    E : array
        2d array defining LHS of DDAE
    A : array
        3d array defining the Ai matrices on the RHS
    hA : array
        1d array defining the delays associated with A
    r : float
        definition of region, either float (half-plane) or list of 4 floats (rectangle)
    **kwargs : 
        discretization (int): discretization for discretizing DDAE into DAE,
            optional, default None, if not specified, heuristic will be used
            to obtain sufficient discretization
        max_size_evp (int): maximum allowed size of eigenvalue problem (EVP)
            optional, default 600
        base_delay (float): base delay in case delays are commensurate,
            optional, default None, used in discretization heuristic
        cd (float): c_D value of provided DDAE, optional, default None, if not
            provided condition c_D < r will be checked in case of RHP region
            and heuristic for obtaining discretization will be envoked if
            condition is not satisfied, if provided, condition c_D < r will be
            checked and warning will be raised if not satisfied. If you want to
            skip this check, provide any value of `cd` smaller then `r`,
            setting `cd=-np.inf` makes sure check will never be performed.

    Returns
    -------
    tuple
        A tuple containing:

        cr : array
            vector of obtained roots
        roots_info : RootsInfo
            RMR metadata containing:
            discretization : int
                discretization used for obtaining EVP 
            gamma_r_exceeds_one : bool
                flag indicating gamma(r) > 1
            index_exceeds_one : bool
                flag that index exceeds one
            discretization_eigenvalues : array
                eigenvalues of EVP
            max_size_evp_enforced : bool
                flag if maximum size of EVP was enforced
            newton_inital_guesses : array
                roots before newton corrections
            newton_final_values : array
                roots after newton corrections
            newton_residuals : array
                newton residuals
            newton_unconverged_initial_guesses : array
                mask of unconverged newton initial guesses
            newton_large_corrections : array
                mask of "large" corrections


    Notes
    -----
    1. assumes non-empty, real E, A, hA
    2. assumes dimensions E.shape[:2] == A.shape[:2] and A.shape[2] == hA.shape[0]
    3. if hA[0] > 0, prepends zero-delay term to hA and A
    4. if region is half-plane and no roots found, warns user
    5. if region is rectangle and no roots found, warns user
    6. uses Newton corrections to refine roots obtained via spectral discretization
    7. if discretization is not provided, heuristic is used to obtain sufficient discretization

    Examples
    --------
    >>> import numpy as np
    >>> from tdspy.stability.characteristic_roots import roots_ddae
    >>> E = np.array([[1,0],[0,1]])
    >>> A = np.zeros(shape=(2,2,2))
    >>> A[:,:,0] = np.array([[0,1],[0,0]])
    >>> A[:,:,1] = np.array([[0,0],[1,0]])
    >>> hA =  np.array([0,1])
    >>> cr, info = roots_ddae(E,A,hA,r=-0.1)
    >>> cr
    array([0.61803399+0.j, -1.61803399+0.j, -0.30901699+0.95105652j,
           -0.30901699-0.95105652j])
    >>> info.discretization
    10
    >>> info.gamma_r_exceeds_one
    False
    >>> info.index_exceeds_one
    False

    """
    # TODO perform checks? This is internal functions -> just list them as
    # comments and implement if necessary
    # 1. non-empty E, A, hA
    # 2. real E, A, hA
    # 3. dimensions E.shape[:2] == A.shape[:2] and A.shape[2] == hA.shape[0]

    # TODO checks for region definition or leave to HIGH level API???

    n = E.shape[0]
    if hA[0] > 0:
        hA = np.r_[0, hA] # prepend 0.0 delay
        A = np.concatenate([np.zeros(shape=(n,n,1), dtype=E.dtype), A], axis=2)

    D, hD = ddae_to_diff(E, A, hA) # use non-compressed form for DIFF
    A, hA = compress_matrices_delays(A, hA)

    # CASE 1: ODE or DAE (no delays)
    if hA.shape[0] == 1:
        logger.debug("CASE: ODE or DAE -> finite number of roots")
        # just use eig and filter based on rhp or region
        roots = linalg.eig(A[:,:,0], E, left=False, right=False)
        roots = roots[np.isfinite(roots)] # get rid of inf and NaN
        if case == "rhp":
            mask = np.isfinite(roots) & (np.real(roots)>=r)
        else: # case == "rect"
            mask = (np.isfinite(roots) & (np.real(roots)>=r[0]) 
                    & (np.real(roots)<=r[1]) & (np.imag(roots)>=r[2])
                    & (np.imag(roots)<=r[3]))
        
        roots_info = RootsInfo(
            discretization=None,
            gamma_r_exceeds_one=False,
            index_exceeds_one=False,
            discretization_eigenvalues=roots.astype(np.complex128),
            max_size_evp_enforced=False,
            newton_inital_guesses=np.zeros((0,), dtype=np.complex128),
            newton_final_values=np.zeros((0,), dtype=np.complex128),
            newton_residuals=np.zeros((0,), dtype=np.float64),
            newton_unconverged_initial_guesses = np.zeros((0,), dtype=bool),
            newton_large_corrections = np.zeros((0,), dtype=bool),
        )
        return roots[mask], roots_info

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
    gamma_r_exceeds_one = None # flag to indicate that gamma(r) > 1, i.e., RHP contains infinitely many roots
    
    if case == "rhp":
        # rescale r
        rs = r * tau_max
        # introduce shift of the origin, shifted matrices B, C
        B = K[:,:,0] + (-rs)*E
        C = K[:,:,1:] * np.exp(-rs * tau_s[1:])
        
        if discretization is None: # no discretization -> envoke heuristic
            logger.debug(f"Discretization not provided, envoking heuristic for RHP with r={r}")
            # step 0: check c_D
            cd = kwargs.get("cd", None)
            if cd is not None: # line 289 - 295, as of now unimportant, later KWARG
                gamma_r_exceeds_one = False
                if cd >= r:
                    logger.warning((f"Condition c_D < r is not satisfied: (c_D={cd} >= r={r}) for user supplied value "
                                    f"of `cd`. This indicates that RHP contains infinitely many roots (assuming "
                                    "provided `cd` is correct and heuristic can not be used)"))
                    gamma_r_exceeds_one = True
            else:
                D, hD = ddae_to_diff(E, A, hA)
                if hD.size != 0: # delay difference equation exists (E is singular)
                    if hD[0] != 0 or np.linalg.matrix_rank(D[0]) < D[0].shape[0]:
                        # first delay is not zero or matrix D0 is not full row rank -> raise value error
                        # TODO: this is original error text from tds-control, should we change it to something like
                        # "provided DDAE is of advanced type" ?
                        raise ValueError("The provided DDAE does not satisfy assumption 2.1.")

                    gamma_val, gamma_info = gamma_diff(D, hD, r)
                    if gamma_val >= 1.0:
                        gamma_r_exceeds_one = True
                        discretization = 30
                        logger.warning((f"gamma({r=}; ...)= {gamma_val} exceeds 1 (i.e., CD > r). Spectral "
                                        "discretization with N = 30 (lowered if maximum size of eigenvalue problem is "
                                        "exceeded). You can (i) provide better `r` (2) provide rectangular region"
                                        "instead of RHP or (3) calculate cd and use r > cd"))
                    else:
                        gamma_r_exceeds_one = False
                
            if discretization is None:
                # discretization is still undefined, reason:
                #   (a) no underlying delay-difference equation or
                #   (b) gamma(r) < 1.0
                #   (c) user knows what he is doing by specifing `cd`
                # => act as region RHP contains finitely many roots and heuristic can be applied
                basic_delay = kwargs.get("base_delay", None)
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
    roots_info = RootsInfo(discretization, False, False, raw_roots, max_size_evp_enforced, newton_roots0, newton_roots, residuals, converged_mask, correction_large_mask)

    return roots, roots_info

def rightmost_root(E: npt.NDArray, A: npt.NDArray, hA: npt.NDArray, r: float=0.0,  **kwargs) -> tuple[complex, RightmostRootInfo]:
    """ Find the rightmost root (RMR) of DDAE represented via E, A, hA

    TDS is assumed to be defined via E, A, hA as

    .. math::
        E \Dot{x}(t) = \sum\limits_{k=1}^N A_k x(t-h_{A,k})

    Parameters
    ----------
        E : array
            2d array defining LHS of DDAE
        A : array
            3d array defining the Ai matrices on the RHS
        hA : array
            1d array defining delays associated with A
        r : float
            definition of complex half-plane, also discretization 
            is perfomed around this point, optional, default 0.0
        **kwargs :
            residual_max : float
                roots with norm=||M(s) @ v|| bellow this
                threshold will be considered as correct solutions, default 1e-6

    Returns
    -------
    tuple

        A tuple containing:

        root_star : complex
            rightmost root
        root_info : RightmostRootInfo
            RMR metadata containing:
            M : array
                characteristic matrix evaluated at `root_star`
            DM : array
                derivative of characteristic matrix evaluated at 
                `root_star`
            u : array
                left eigenvector associated with `root_star`
            v : array
                right eigenvector associated with `root_star`
            found : bool
                flag if RMR succesfully found
            max_size_evp_enforced : bool
                flag if maximum size of EVP was enforced

    """
    # unpack kwargs
    residual_max = kwargs.get("residual_max", 1e-6)
    
    # TODO perform checks? This is internal functions -> just list them as
    # comments and implement if necessary
    # 1. non-empty E, A, hA
    # 2. real E, A, hA
    # 3. dimensions E.shape[:2] == A.shape[:2] and A.shape[2] == hA.shape[0]
    # 4. residual_max is positive float

    # find all roots via discretization
    cr, cr_info = roots_ddae(E, A, hA, r)

    # find right most root
    root_star = None
    root_star_found = False # indicator, that right most root succesfully found

    # CASE 1: roots cr are empty
    if cr.size == 0:
        logger.debug(f"No roots in RHP Re(z) >= {r}, using eigenvalues obtained via discretization")
        # try to use eigenvalues of discretized DDAE
        u = np.sort_complex(cr_info.discretization_eigenvalues) # in ascending order

        # check whether the approximations obtained using spectraldiscretisation
        # make the characteristic matrix sufficiently close to being singular.
        for root in u[::-1]: # iterate from root with largest Re(root)
            # evaluate characteristic matrix
            M = root*E + np.sum(A * np.exp(-hA*root), axis=2)

            # perform SVD and check what is the smallest eigen value
            U, s, Vh = linalg.svd(M, compute_uv=True) # sorted in non-increasing order
            v0 = np.conj(Vh[-1]) # eigen vector associated to smallest eigenvalue
            residual = linalg.norm(M @ v0[:, np.newaxis], ord=None, axis=None) # 2-norm of np.ravel(.) is returned
            
            # if residual ||M(s) @ v0|| sufficiently close to 0 -> RMR found
            if residual <= residual_max:
                root_star = s[-1]
                root_star_found = True
                break
    
        if root_star is None: # no root to the right of r found
            u_smaller_r = u[np.real(u) < r]
            if u_smaller_r.size == 0: # worst case scenario, rutrn fallback value
                root_star = r + 0j
                logger.warning(f"No RMR found, returning {root_star=} (=r) as a fallback value")
            else: # take the rightmost root to the left of r
                root_star = u_smaller_r[-1] # already sorted, take last element

    # CASE 2: the rightmost point in cr lies significantly to the left of
    # the rightmost point eigen value from discretization
    elif np.max(np.real(cr)) <= lower_bound(np.max(np.real(cr_info.discretization_eigenvalues)), 0.05, 1.e-3):
        logger.debug(f"RMR lies significantly to the left od RMR of discretization EVP solution.")
        cr_discretization = cr_info.discretization_eigenvalues
        imax = np.argmax(np.real(cr_discretization))
        root_star1 = cr_discretization[imax]

        # check the roots possibly right of root_star1
        u = np.vectorize(upper_bound)(np.real(cr_discretization), 0.05, 1.e-3)
        u = np.sort_complex(u)

        # check whether the approximations obtained using spectraldiscretisation
        # make the characteristic matrix sufficiently close to being singular.
        for root in u[::-1]: # iterate from root with largest Re(root)
            # evaluate characteristic matrix
            M = root*E + np.sum(A * np.exp(-hA*root), axis=2)

            # perform SVD and check what is the smallest eigen value
            U, s, Vh = linalg.svd(M, compute_uv=True) # sorted in non-increasing order
            v0 = np.conj(Vh[-1]) # eigen vector associated to smallest eigenvalue
            residual = linalg.norm(M @ v0[:, np.newaxis], ord=None, axis=None) # 2-norm of np.ravel(.) is returned

            # if residual ||M(s) @ v0|| sufficiently close to 0 -> RMR found
            if residual <= residual_max:
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