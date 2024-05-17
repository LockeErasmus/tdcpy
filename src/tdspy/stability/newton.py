"""
Newton method for increasing precision of roots
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg


logger = logging.getLogger(__name__)

def newton_correction(roots0: npt.NDArray, E: npt.NDArray, A: npt.NDArray,
                      hA: npt.NDArray, inplace=False, **kwargs):
    """ Newton corrections applied initial guess `roots0`


    root `r` (element of `roots0`) is an initial guess of the eigenvalue of
    characteristic matrix

        M(r) = E*r - A[0]*exp(-r*hA[0]) + ... + A[mA]*exp(-r*hA[mA])
    
    and we look for better solution of system

        M(r) * v(r) - r*v(r) = 0
        v(r0)' * v(r) - 1    = 0
    
    where vector of variables x = [v, r].

    Args:
        ... TODO
        **kwargs:
            tol (float): absolute tolerance, default 1e-10
            max_iterations(int): maximum number of newton iterations, default 20

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
    if inplace:
        roots = roots0
    else:
        roots = np.copy(roots0)

    max_iterations = kwargs.get("max_iterations", 20)
    tol = kwargs.get("tol", 1e-8)
    n = E.shape[0]

    # TODO pre-allocate memory
    M = np.zeros(shape=(n, n), dtype=roots.dtype)
    dM = np.zeros(shape=(n, n), dtype=roots.dtype)
    v0 = np.zeros(shape=(n,), dtype=roots.dtype)
    jacobian = np.zeros(shape=(n+1, n+1), dtype=roots.dtype)
    v = np.zeros_like(v0, dtype=roots.dtype)
    f_val = np.zeros(shape=(n+1,), dtype=roots.dtype)

    for i in range(roots.shape[0]):
        # initial guess eigen value `roots[i]`
        # evaluate characteristic matrix M(roots[i])
        M[:,:] = (roots[i] * E - A[:,:,0] - np.sum(A[:,:,1:]*np.exp(-roots[i]*hA[1:]), axis=2))
        (_, _, Vh) = linalg.svd(M, compute_uv=True) # (U, s, Vh) = linalg.svd(...)
        v0[:] = Vh[-1] # initial right null eigen-vector
        residual = linalg.norm(M @ v0[:, np.newaxis], ord=None, axis=None) # 2-norm of np.ravel(.) is returned
                
        logger.debug(f"NEWTON | initial {residual=}, root={roots[i]}")
        logger.debug(f"---------------------------------------------")
        if residual <= tol:
            #continue
            pass
        else:
            # jacobian[-1, -1] = 0 <- always fullfilled as jacobian initilized as zero matrix
            v[:] = v0 # initial v <- v0
            for j in range(max_iterations):
                
                dM[:,:] = E + np.sum(A[:,:,1:]*(hA[1:]*np.exp(-roots[i]*hA[1:])), axis=2)
                # update jacobian
                jacobian[:n, :n] = M
                jacobian[:n, -1] = (dM @ v[:, np.newaxis])[:,0] # TODO check if this is efficient
                jacobian[-1, :n] = 2*np.conj(v)
                # update value of functions
                f_val[:n] = (-M @ v[:, np.newaxis])[:,0]
                f_val[-1] = -(np.inner(np.conj(v), v) - 1)
                
                (dx, _, _, _) = linalg.lstsq(jacobian, f_val[:,np.newaxis])
                #dx = linalg.inv(jacobian)@f_val[:,np.newaxis]
                
                # update roots and right eigen-vector
                roots[i] += dx[-1, 0]
                v += dx[:n, 0]

                # update M
                M[:,:] = (roots[i] * E - A[:,:,0]
                    - np.sum(A[:,:,1:]*np.exp(-roots[i]*hA[1:]), axis=2))
                
                # check if converged
                residual = linalg.norm(M @ v[:, np.newaxis], ord=None, axis=None)
                logger.debug(f"  it: {j} | initial {residual=}, root={roots[i]}")
                if residual <= tol:
                    break # converged
            
            if residual > tol:
                logger.warning(f"The Newton corrections failed to converge for root={roots0[i]}, final {residual=}")

    
    return roots # TODO, return None if inplace``

def newton_correction_(roots0: npt.NDArray, E: npt.NDArray, A: npt.NDArray,
                      hA: npt.NDArray, inplace=False, **kwargs):
    """ Newton corrections applied initial guess `roots0`


    root `r` (element of `roots0`) is an initial guess of the eigenvalue of
    characteristic matrix

        M(r) = E*r - A[0]*exp(-r*hA[0]) + ... + A[mA]*exp(-r*hA[mA])
    
    and we look for better solution of system

        M(r) * v(r) - r*v(r) = 0
        v(r0)' * v(r) - 1    = 0
    
    where vector of variables x = [v, r].

    Args:
        ... TODO
        **kwargs:
            tol (float): absolute tolerance, default 1e-10
            max_iterations(int): maximum number of newton iterations, default 20

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
    if inplace:
        roots = roots0
    else:
        roots = np.copy(roots0)

    max_iterations = kwargs.get("max_iterations", 20)
    tol = kwargs.get("tol", 1e-8)
    n = E.shape[0]

    # TODO pre-allocate memory
    M = np.zeros(shape=(n, n), dtype=roots.dtype)
    dM = np.zeros(shape=(n, n), dtype=roots.dtype)
    v0 = np.zeros(shape=(n,), dtype=roots.dtype)
    jacobian = np.zeros(shape=(n+1, n+1), dtype=roots.dtype)
    v = np.zeros_like(v0, dtype=roots.dtype)
    f_val = np.zeros(shape=(n+1,), dtype=roots.dtype)

    for i in range(roots.shape[0]):
        # initial guess eigen value `roots[i]`
        # evaluate characteristic matrix M(roots[i])
        M[:,:] = (roots[i] * E - A[:,:,0] - np.sum(A[:,:,1:]*np.exp(-roots[i]*hA[1:]), axis=2))
        (_, _, Vh) = linalg.svd(M, compute_uv=True) # (U, s, Vh) = linalg.svd(...)
        v0[:] = Vh[-1] # initial right null eigen-vector
        residual = linalg.norm(M @ v0[:, np.newaxis], ord=None, axis=None) # 2-norm of np.ravel(.) is returned
                
        logger.debug(f"NEWTON | initial {residual=}, root={roots[i]}")
        logger.debug(f"---------------------------------------------")
        if residual <= tol:
            #continue
            pass
        else:
            jacobian[-1, :n] = np.conj(v0)
            # jacobian[-1, -1] = 0 TODO delete, it is already 0.0
            v[:] = v0 # initial v <- v0
            for j in range(max_iterations):
                
                dM[:,:] = E + np.sum(A[:,:,1:]*(hA[1:]*np.exp(-roots[i]*hA[1:])), axis=2)
                # update jacobian
                jacobian[:n, :n] = M
                jacobian[:n, -1] = (dM @ v[:, np.newaxis])[:,0] # TODO check if this is efficient
                # update value of functions
                f_val[:n] = (-M @ v[:, np.newaxis])[:,0]
                f_val[-1] = -(np.inner(jacobian[-1, :n], v) - 1) # np.conj(v0) is already stored in jacobian[-1, :n]
                
                (dx, _, _, _) = linalg.lstsq(jacobian, f_val[:,np.newaxis])
                #dx = linalg.inv(jacobian)@f_val[:,np.newaxis]
                
                # update roots and right eigen-vector
                roots[i] += dx[-1, 0]
                v += dx[:n, 0]

                # update R_lambda
                M[:,:] = (roots[i] * E - A[:,:,0]
                    - np.sum(A[:,:,1:]*np.exp(-roots[i]*hA[1:]), axis=2))
                
                # check if converged
                residual = linalg.norm(M @ v[:, np.newaxis], ord=None, axis=None)
                logger.debug(f"  it: {j} | initial {residual=}, root={roots[i]}")
                if residual <= tol:
                    break # converged
            
            if residual > tol:
                logger.warning(f"The Newton corrections failed to converge for root={roots0[i]}, final {residual=}")

    
    return roots # TODO, return None if inplace``

    
    

