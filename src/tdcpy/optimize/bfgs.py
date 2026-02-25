import numpy as np
from numpy.linalg import norm
from scipy.optimize import OptimizeResult


# ============================================================
# Weak Wolfe line search
# ============================================================

def _weak_wolfe_line_search(fg, x, p, f0, g0,
                            c1=1e-4, c2=0.1,
                            alpha0=1.0,
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

        # Armijo
        if f_new > f0 + c1 * alpha * g0_dot_p:
            alpha_high = alpha
            alpha = 0.5 * (alpha_low + alpha_high)
            continue

        # Weak curvature
        if np.dot(g_new, p) < c2 * g0_dot_p:
            alpha_low = alpha
            if np.isinf(alpha_high):
                alpha *= 2.0
            else:
                alpha = 0.5 * (alpha_low + alpha_high)
            continue

        return alpha, f_new, g_new

    return alpha, f_new, g_new


# ============================================================
# SciPy-compatible solver
# ============================================================

def bfgs_weak_wolfe(fun, x0, args=(), jac=None, callback=None,
                    maxiter=None, tol=None, disp=False,
                    return_all=False, **options):

    x = np.asarray(x0, dtype=float)
    n = len(x)

    maxiter = maxiter or options.get("maxiter", 200)
    tol = tol or options.get("gtol", 1e-6)

    c1 = options.get("c1", 1e-4)
    c2 = options.get("c2", 0.1)
    alpha0 = options.get("alpha0", 1.0)

    # --------------------------------------------------------
    # Handle jac options (SciPy-style)
    # --------------------------------------------------------

    if jac is True:
        # fun returns (f, g)
        def fg(z):
            return fun(z, *args)

    elif callable(jac):
        def fg(z):
            return fun(z, *args), jac(z, *args)

    else:
        raise ValueError("This solver requires jac=True or jac=callable")

    # --------------------------------------------------------

    H = np.eye(n)

    f, g = fg(x)

    nfev = 1
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

        # OPTION 1:
        # Curvature safeguard - update Hessian only if:
        # sy = np.dot(s, y)
        # if sy > 1e-8 * norm(s) * norm(y):
        #     rho = 1.0 / sy
        #     I = np.eye(n)
        #     H = (I - rho * np.outer(s, y)) @ H @ \
        #         (I - rho * np.outer(y, s)) + \
        #         rho * np.outer(s, s)
            
        # OPTION 2:
        # POWELL DAMPING to maintain positive-definite H
        sHs = np.dot(s, H @ s)
        sy = np.dot(s, y)
        if sy < 1e-8 * norm(s) * norm(y):
            # option 1 just fix to 0.8
            # theta = 0.8 # How much I trust observed gradient versus Hesssion
            # prediction, has to be from [0,1], common seems to be [0.5, 0.95]

            # option 2 automatic theta:
            # compute theta automatically
            theta = (0.8 * sHs) / (sHs - sy) if (sHs - sy) != 0 else 1.0
            theta = min(max(theta, 0.0), 1.0)
            y = theta * y + (1 - theta) * (H @ s) # replace y wit y tilde ...
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
            print(f"iter {k:3d} | f = {f:.6e} | ||g|| = {norm(g):.3e} {x} \n  Hk  cond={np.linalg.cond(H)}\n{H}")

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