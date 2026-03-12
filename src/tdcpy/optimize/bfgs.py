from typing import Callable

import numpy as np
import numpy.typing as npt
from numpy.linalg import norm
from scipy.optimize import OptimizeResult


class BFGSInverse:
    """ Dense BFGS update class """
    def __init__(self, n, scale=1.0):
        self.n = n
        self.I = np.eye(n)
        self.B = scale * self.I
        
    def apply(self, V: npt.NDArray):
        """ Apply B (approximation of hessian inverse H^{-1})"""
        V = np.asarray(V)
        return self.B @ V
    
    def update(self, s: npt.NDArray, y: npt.NDArray, tol: float=1e-12):
        """ Dense update """
        s = s.flatten()
        y = y.flatten()

        ys = y @ s
    
        if ys <= tol:
            return
        
        s_col = s.reshape(-1, 1)
        y_col = y.reshape(-1, 1)

        rho = 1.0 / ys
        Vmat = self.I - rho * s_col @ y_col.T
        self.B = Vmat @ self.B @ Vmat.T + rho * s_col @ s_col.T

class LBFGSInverse:
    """ Limited memory version of BFGSInverse """

class BFGSInverse:
    """
    Hybrid BFGS inverse:
    - Dense mode (memory=None)
    - L-BFGS mode (memory=m)
    """

    def __init__(self, n, scale=1.0, memory: int|None=None):
        self.n = n
        self.memory = memory

        if memory is None:
            # Dense mode
            self.B = scale * np.eye(n)
        else:
            # L-BFGS mode
            self.s_list = []
            self.y_list = []
            self.rho_list = []
            self.gamma = scale

    # ==========================
    # Apply H^{-1}
    # ==========================

    def apply(self, V):
        V = np.asarray(V)

        if self.memory is None:
            return self.B @ V

        # L-BFGS two-loop recursion
        if V.ndim == 1:
            return self._two_loop(V)
        elif V.ndim == 2:
            return np.column_stack([self._two_loop(V[:, i])
                                    for i in range(V.shape[1])])
        else:
            raise ValueError("Input must be vector or 2D matrix.")

    def _two_loop(self, q):
        q = q.copy()
        alpha = []

        # First loop
        for s, y, rho in reversed(list(zip(
                self.s_list, self.y_list, self.rho_list))):
            a = rho * (s @ q)
            alpha.append(a)
            q -= a * y

        # Initial scaling
        r = self.gamma * q

        # Second loop
        for s, y, rho, a in zip(
                self.s_list, self.y_list,
                self.rho_list, reversed(alpha)):
            b = rho * (y @ r)
            r += s * (a - b)

        return r

    # ==========================
    # Update
    # ==========================

    def update(self, s, y, damping=True):

        s = s.flatten()
        y = y.flatten()

        ys = y @ s
        if ys <= 1e-12:
            return

        if self.memory is None:
            # Dense update
            s_col = s.reshape(-1, 1)
            y_col = y.reshape(-1, 1)

            rho = 1.0 / ys
            I = np.eye(self.n)
            Vmat = I - rho * s_col @ y_col.T
            self.B = Vmat @ self.B @ Vmat.T + rho * s_col @ s_col.T

        else:
            # L-BFGS update
            rho = 1.0 / ys

            if len(self.s_list) == self.memory:
                self.s_list.pop(0)
                self.y_list.pop(0)
                self.rho_list.pop(0)

            self.s_list.append(s)
            self.y_list.append(y)
            self.rho_list.append(rho)

            # Update scaling
            self.gamma = ys / (y @ y)




def _weak_wolfe_line_search(fg: Callable, x: npt.NDArray, p, f0, g0, c1=1e-4, c2=0.1, alpha0=1.0,
                            max_iter=40):

    alpha = alpha0
    g0_dot_p = np.dot(g0, p)

    # if g0_dot_p >= 0:
    #     raise ValueError("Search direction is not descent")
    if g0_dot_p >= 0:
        # fallback to steepest descent along -g0
        p = -g0
        g0_dot_p = np.dot(g0, p)
        alpha = min(alpha0, 0.1)  # TODO safer initial step

    alpha_low = 0.0
    alpha_high = np.inf

    for _ in range(max_iter):
        x_new = x + alpha * p
        f_new, g_new = fg(x_new)

        # Weak Wolfe conditions
        ## Armijo (aka sufficient decrease)
        if f_new > f0 + c1 * alpha * g0_dot_p:
            alpha_high = alpha
            alpha = 0.5 * (alpha_low + alpha_high)
            continue

        # Curvature condition
        if np.dot(g_new, p) < c2 * g0_dot_p:
            alpha_low = alpha
            if np.isinf(alpha_high):
                alpha *= 2.0
            else:
                alpha = 0.5 * (alpha_low + alpha_high)
            continue

        return alpha, f_new, g_new
    
    # It is possible at this point, that no step (alpha) is selected
    # TODO: just ignore Curvature condition and run only with Armijo?
    # logger.warning("Weak wolfe line search failed")
    return alpha, f_new, g_new

def bfgs_weak_wolfe(fun, x0, args=(), jac=None, callback=None,
                    maxiter=None, tol=None, disp=False,
                    return_all=False, **options):
    """ Scipy compatible BFGS with weak Wolfe line search
    
    """

    x = np.asarray(x0, dtype=float)
    n = len(x)

    maxiter = maxiter or options.get("maxiter", 200)
    tol = tol or options.get("gtol", 1e-6)

    c1 = options.get("c1", 1e-4)
    c2 = options.get("c2", 0.1)
    alpha0 = options.get("alpha0", 1.0)

    # I have added this part to handle jac=None|Callable|True like in scipy
    if jac is True: # fun is implemented such that it also returns jac
        def fg(z):
            return fun(z, *args)

    elif callable(jac):
        def fg(z):
            return fun(z, *args), jac(z, *args)

    else: # TODO: use numerical derivative? or just raise?
        raise ValueError("This solver requires jac=True or jac=callable")

    H = np.eye(n)

    f, g = fg(x)

    nfev = 1 # TODO: as of now not properly working
    njev = 1

    allvecs = [x.copy()] if return_all else None

    k = 0
    success = False

    while k < maxiter:

        if norm(g) < tol:
            success = True
            break
        
        # DO this every 10th iterations??
        # cond_H = np.linalg.cond(H)
        # if cond_H > 1e8:
        #     print("reseting condition number")
        #     H = np.eye(n) * np.trace(H)/n  # reset to scaled identity

        p = -H @ g

        # normalize direction
        max_step = 1.0
        if norm(p) > max_step:
            p = p / norm(p) * max_step
        
        # If the direction points uphill, just switch sign
        if np.dot(g, p) >= 0:
            p = -g

        alpha, f_new, g_new = _weak_wolfe_line_search(
            fg, x, p, f, g,
            c1=c1, c2=c2,
            alpha0=alpha0
        )

        nfev += 1
        njev += 1

        s = alpha * p
        x_new = x + s
        y = g_new - g

        # OPTION 1: Curvature safeguard - update Hessian only if it makes sense
        # sy = np.dot(s, y)
        # if sy > 1e-8 * norm(s) * norm(y):
        #     rho = 1.0 / sy
        #     I = np.eye(n)
        #     H = (I - rho * np.outer(s, y)) @ H @ \
        #         (I - rho * np.outer(y, s)) + \
        #         rho * np.outer(s, s)
            
        # OPTION 2: Powell damping to maintain positive-definite H
        sHs = np.dot(s, H @ s)
        sy = np.dot(s, y)
        if sy < 1e-8 * norm(s) * norm(y):
            # suboption 1: I can just fix to theta to 0.8
            # theta = 0.8 # How much I trust observed gradient versus Hesssion
            # prediction, has to be from [0,1], common seems to be [0.5, 0.95]

            # option 2 automatic theta:
            # compute theta automatically
            theta = (0.8 * sHs) / (sHs - sy) if (sHs - sy) != 0 else 1.0
            theta = min(max(theta, 0.0), 1.0)
            y = theta * y + (1 - theta) * (H @ s) # replace y with y tilde
            sy = np.dot(s, y)

        if sy > 1e-12:
            rho = 1.0 / sy
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x, f, g = x_new, f_new, g_new

        if return_all:
            allvecs.append(x.copy())

        if callback is not None:
            callback(x)

        if disp:
            print(f"iter {k:3d} | f = {f:.6e} | ||g|| = {norm(g):.3e}")

        k += 1

    result = OptimizeResult(
        x=x,
        fun=f,
        jac=g,
        hess_inv=H,
        nfev=nfev,
        njev=njev,
        nit=k,
        success=success,
        message="Converged" if success else "Maximum iterations exceeded"
    )

    if return_all:
        result.allvecs = allvecs

    return result