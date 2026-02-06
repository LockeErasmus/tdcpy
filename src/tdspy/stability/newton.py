"""
Newton method for increasing precision of roots
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg


logger = logging.getLogger(__name__)

def newton_correction(roots0: npt.NDArray, E: npt.NDArray, A: npt.NDArray,
                      hA: npt.NDArray, **kwargs):
    """ Newton corrections applied initial guess `roots0`

    root `s` (element of `roots0`) is an initial guess of the eigenvalue of
    characteristic matrix M(s) and its derivative dM(s)

        M(s) = E*s - A[0]*exp(-s*hA[0]) - ... - A[mA]*exp(-s*hA[mA])
        dM(s) = E + hA[0]*A[0]*exp(-s*hA[0]) + ... + hA[mA]*A[mA]*exp(-s*hA[mA])
    
    and we look for a solution of system n+1 non-linear equations

        M(s) * v      ==  0
        v0.H * v - 1  ==  0
    
    with jacobian

    J = [[M(s), dM(s)*v]
         [v0.H, 0      ]]

    Update rule for newton corrections stands as

        [dv.T, ds].T = PINV( J ) @ [M(s)*v, v0.H * v - 1].T
        
        v = v - dv
        s = s - ds

    Args:
        roots0 (array): 1D vector of initial guesses for roots
        E (array): E matrix from TDS representation shaped (n,n)
        A (array): A matrices from TDS representation shaped (n,n,mA)
        hA (array): hA vector of delays from TDS representation shaped (mA,)
        **kwargs:
            return_residuals(bool): wheter to also return residuals or not,
                default False
            tol (float): absolute tolerance, default 1e-10
            max_iterations(int): maximum number of newton iterations, default 20
    
    Returns:
        tuple containing:

            - roots (array): 1D array roots of improved precission
            - residuals (array): 1D array roots of residuals
            - converged_mask (array): 1D array mask if root converged
            - large_correction_mask (array): 1D array mask if correction large

    Notes:
        1. The expected shapes of input arrays:
            (a) E ... (n,n)
            (b) A ... (n,n,mA)
            (c) hA ... (mA,)
         shapes are not checked to speed up computation.
        2. It is possible some corrections won't converge (`residual` check)
        3. It is possible some corrections converge, hoewever corrections are
          "large" (see "newton method - basin of attraction"), these corrections
          should be taken with a grain of salt because root can converge to root
          which is already present in solution.
    """
    return_residuals = kwargs.get("return_residuals", False)
    max_iterations = kwargs.get("max_iterations", 20)
    tol = kwargs.get("tol", 1e-10)

    roots = np.copy(roots0)
    
    # pre-allocate memory
    n = E.shape[0]
    M = np.zeros(shape=(n, n), dtype=roots.dtype)
    dM = np.zeros(shape=(n, n), dtype=roots.dtype)
    v0 = np.zeros(shape=(n,), dtype=roots.dtype)
    jacobian = np.zeros(shape=(n+1, n+1), dtype=roots.dtype)
    v = np.zeros_like(v0, dtype=roots.dtype)
    f_val = np.zeros(shape=(n+1,), dtype=roots.dtype)
    residuals = np.zeros(shape=roots0.shape, dtype=np.float64)
    converged_mask = np.full_like(roots, fill_value=True, dtype=bool)
    large_correction_mask = np.full_like(roots, fill_value=False, dtype=bool)

    for i in range(roots.shape[0]):
        # initial guess of eigen-value stored in `roots[i]`
        # initial guess of right-null vector stored in `v0`
        # evaluate characteristic matrix M(roots[i]) to obtain v0
        M[:,:] = (roots[i] * E - A[:,:,0] - np.sum(A[:,:,1:]*np.exp(-roots[i]*hA[1:]), axis=2))
        (_, _, Vh) = linalg.svd(M, compute_uv=True)
        v0[:] = np.conj(Vh[-1])
        residuals[i] = linalg.norm(M @ v0[:, np.newaxis], ord=None, axis=None) # 2-norm of np.ravel(.) is returned

        if residuals[i] <= tol:
            logger.debug(f"Root: s0={roots[i]} already fullfils ||M(s0)@v0||={residuals[i]} <= tol={tol}")
        elif np.any(np.real(roots[i]*hA[1:]) < -700 ): # bound is np.log(np.finfo(np.float64).max) = 709.782712893384
            logger.warning(f"Evaluating of EXP( -s*tau ) would result in overflow. Stopping iterations for root: s0={roots[i]} and skipping newton corrections.")
            large_correction_mask[i] = True
            converged_mask[i] = False
            residuals[i] = np.inf
        else:
            logger.debug(f"Envoking Newton corrections for root: s0={roots[i]} ||M(s0)@ v0||={residuals[i]} > tol={tol}")
            v[:] = v0 # initial v <- v0
            for j in range(max_iterations):
                logger.debug(f"  it: {j} | residual={residuals[i]}, root={roots[i]}")   
                dM[:,:] = E + np.sum(A[:,:,1:]*(hA[1:]*np.exp(-roots[i]*hA[1:])), axis=2)               
                
                # update jacobian
                jacobian[:n, :n] = M
                jacobian[:n, -1] = (dM @ v[:, np.newaxis])[:,0]
                jacobian[-1, :n] = np.conj(v0)
                # jacobian[-1, -1] = 0 <- always fullfilled as jacobian initilized as zero matrix
                
                # update value of functions
                f_val[:n] = (M @ v[:, np.newaxis])[:,0]
                f_val[-1] = (np.inner(np.conj(v0), v) - 1)

                # obtain update `dx = [dv.T, ds].T ` by solving LS problem
                (dx, _, _, _) = linalg.lstsq(jacobian, f_val[:,np.newaxis])
                
                
                # update roots and right eigen-vector
                v -= dx[:n, 0]
                roots[i] -= dx[-1, 0]

                if np.any(np.real(roots[i]*hA[1:]) < -700 ): # bound is np.log(np.finfo(np.float64).max) = 709.782712893384
                    logger.warning(f"Evaluating of EXP( -s*tau ) would result in overflow. Stopping iterations for root: s0={roots[i]} and skipping newton corrections.")
                    large_correction_mask[i] = True
                    residuals[i] = np.inf
                    break # stop newton iterations for this root

                # update characteristic matrix M
                M[:,:] = (roots[i] * E - A[:,:,0] - np.sum(A[:,:,1:]*np.exp(-roots[i]*hA[1:]), axis=2))

                # check if residual <= desired tolerance
                residuals[i] = linalg.norm(M @ v[:, np.newaxis], ord=None, axis=None)
                # logger.debug(f"  it: {j} | residual={residuals[i]}, root={roots[i]}")
                if residuals[i] <= tol: # converged
                    # check for large newton correction
                    if linalg.norm(v, ord=None, axis=None) > tol and np.abs(roots0[i] - roots[i]) / np.max([(np.abs(roots0[i]) + np.abs(roots[i]))/2,1]) > 0.1:
                        large_correction_mask[i] = True
                        logger.warning(f"Large Newton corrections for {roots0[i]} (New value: {roots[i]})")
                    break # converged
            
            if residuals[i] > tol:
                converged_mask[i] = False
                logger.warning(f"The Newton corrections failed to converge for root={roots0[i]}, final residual {residuals[i]}")

    return roots, residuals, converged_mask, large_correction_mask
