"""
Newton method for increasing precision of roots
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg


logger = logging.getLogger(__name__)

def newton_correction(roots0: npt.NDArray, E: npt.NDArray, A: npt.NDArray,
                      hA: npt.NDArray, inplace: bool=False, **kwargs):
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
        inplace (bool): whether to apply newton corrections to input vector
                or create copy and do not alter initial roots, default False
        **kwargs:
            return_residuals(bool): wheter to also return residuals or not,
                default False
            tol (float): absolute tolerance, default 1e-10
            max_iterations(int): maximum number of newton iterations, default 20
    
    Returns:
        None : if inplace=True and return_residuals=False
        roots (array): if inplace=False and return residuals=False
        residuals (array): if inplace=True and return_residuals=True
        (roots, residuals) (tuple): if inplace=False and return_residuals=True

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

    if inplace:
        roots = roots0
    else:
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
        else:
            logger.debug(f"Envoking Newton corrections for root: s0={roots[i]} ||M(s0)@ v0||={residuals[i]} > tol={tol}")
            v[:] = v0 # initial v <- v0
            for j in range(max_iterations):   
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

                # update characteristic matrix M
                M[:,:] = (roots[i] * E - A[:,:,0] - np.sum(A[:,:,1:]*np.exp(-roots[i]*hA[1:]), axis=2))
                
                # check if residual <= desired tolerance
                residuals[i] = linalg.norm(M @ v[:, np.newaxis], ord=None, axis=None)
                logger.debug(f"  it: {j} | residual={residuals[i]}, root={roots[i]}")
                if residuals[i] <= tol:
                    break # converged
            
            if residuals[i] > tol:
                logger.warning(f"The Newton corrections failed to converge for root={roots0[i]}, final residual {residuals[i]}")

    # return logic: TODO this is a little bit messy, but to match desired behaviour ...
    if inplace and not return_residuals:
        return None
    elif inplace and return_residuals:
        return residuals  
    elif not inplace and return_residuals:
        return roots, residuals
    else:
        return roots
