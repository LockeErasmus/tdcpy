"""
Discretization heuristic
------------------------

References:

 [1] Wu, Z., & Michiels, W. Reliably computing all characteristic roots of
     delay differential equations in a given right half plane using a 
     spectral method. Journal of Computational and Applied Mathematics,
     236(9), 2012, pp. 2499-2514.

"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg, interpolate

logger = logging.getLogger(__name__)

# coefficients of the functions a(theta) and b(theta) that are used to
# approximate R(N;theta) (which on itself defines the region in which the
# exponential function is sufficiently well approximated). More specifically,
# R(N;theta) ≈ b(theta) + a(theta)*N.
B_THETA: npt.NDArray = np.array([-0.2462, -0.2266, -0.2053, -0.2302, -0.2326, 
                                 -0.2335, -0.2362, -0.2421, -0.2463, -0.2604,
                                 -0.2656, -0.2749, -0.2919, -0.3030, -0.3140,
                                 -0.3265, -0.3491, -0.3664, -0.3892, -0.4204,
                                 -0.4548, -0.4855, -0.5339, -0.5872, -0.6491,
                                 -0.7354, -0.8478, -0.9930, -1.1800, -1.4448,
                                 -1.9414, -2.7149, -3.0292], dtype=np.float64)
A_THETA: npt.NDArray = np.array([0.9124, 0.9123, 0.9136, 0.9165, 0.9195, 0.9234,
                                 0.9285, 0.9345, 0.9416, 0.9501, 0.9592, 0.9698,
                                 0.9818, 0.9947, 1.0090, 1.0249, 1.0427, 1.0620,
                                 1.0833, 1.1069, 1.1331, 1.1614, 1.1936, 1.2289,
                                 1.2685, 1.3132, 1.3642, 1.4231, 1.4913, 1.5731,
                                 1.6783, 1.7867, 1.8183], dtype=np.float64)
THETA: npt.NDArray = np.linspace(0, np.pi/4, 33, dtype=np.float64)

cubic_spline_a = interpolate.CubicSpline(THETA, A_THETA)
cubic_spline_b = interpolate.CubicSpline(THETA, B_THETA)

def commensurate_gk(E, B, C, n_k, grid_points=20) -> npt.NDArray:
    """ TODO """
    stepsize = np.pi / grid_points
    #factor = 1.05*np.sin(stepsize)
    gk = []
    jhh = np.pi / (grid_points*n_k[-1])
    for k in range(grid_points*n_k[-1]):
        coef = np.exp(1j*k*jhh*n_k[1:])
        W = B + np.sum(C*coef, axis=2)
        r = linalg.eig(W, E, left=False, right=False)
        gk.append(np.conjugate(r)) # complex conjugate
    return np.concatenate(gk)

def commensurate_gk2(E, B, C, tau, n_k, si, grid_points=20) -> npt.NDArray:
    """ TODO """
    stepsize = np.pi / grid_points
    factor = 1.05*np.sin(stepsize)
    gk = []
    jhh = np.pi / (grid_points*n_k[-1])
    for k in range(grid_points*n_k[-1]):
        coef = np.exp(1j*k*jhh*n_k[1:]) * np.exp(-factor * si * tau[1:])
        W = B + np.sum(C*coef, axis=2)
        r = linalg.eig(W, E, left=False, right=False)
        gk.append(np.conjugate(r)) # complex conjugate
    return np.concatenate(gk)

def disproportionate_gk(E, B, C, tau, grid_points=20) -> npt.NDArray:
    """ TODO - in original implementation, only 3 non zero delays allowed """
    n_delays = len(tau)
    stepsize = np.pi / grid_points

    gk = []
    if n_delays == 2:
        for k in range(grid_points + 1):
            W = B + C[:,:,0] * np.exp(1j*k*stepsize)
            r = linalg.eig(W, E, left=False, right=False)
            gk.append(np.conjugate(r))
    elif n_delays == 3:
        for k in range(grid_points + 1):
            for j in range(-grid_points+1, grid_points+1, 1):
                W = B + C[:,:,0] * np.exp(1j*k*stepsize) + C[:,:,1] * np.exp(1j*j*stepsize)
                r = linalg.eig(W, E, left=False, right=False)
                gk.append(np.conjugate(r))
    elif n_delays == 4:
        for k in range(grid_points + 1):
            for j in range(-grid_points+1, grid_points+1, 1):
                for i in range(-grid_points+1, grid_points+1, 1):
                    W = B + C[:,:,0] * np.exp(1j*k*stepsize) + C[:,:,1] * np.exp(1j*j*stepsize) + C[:,:,2] * np.exp(1j*i*stepsize)
                    r = linalg.eig(W, E, left=False, right=False)
                    gk.append(np.conjugate(r))
    else:
        raise NotImplementedError(f"Not implemented for more than 3 non-zero delays.")
    return np.concatenate(gk)

def disproportionate_gk2(E, B, C, tau, si, grid_points=20) -> npt.NDArray:
    """ TODO - in original implementation, only 3 non zero delays allowed """
    n_delays = len(tau)
    stepsize = np.pi / grid_points
    factor = 1.05*np.sin(stepsize)
    gk = []
    if n_delays == 2:
        for k in range(grid_points + 1):
            W = B + C[:,:,0] * np.exp(1j*k*stepsize) * np.exp(-factor*si*tau[1])
            r = linalg.eig(W, E, left=False, right=False)
            gk.append(np.conjugate(r))
    elif n_delays == 3:
        for k in range(grid_points + 1):
            for j in range(-grid_points+1, grid_points+1, 1):
                W = (B
                     + C[:,:,0] * np.exp(1j*k*stepsize) * np.exp(-factor*si*tau[1])
                     + C[:,:,1] * np.exp(1j*j*stepsize) * np.exp(-factor*si*tau[2]))
                r = linalg.eig(W, E, left=False, right=False)
                gk.append(np.conjugate(r))
    elif n_delays == 4:
        for k in range(grid_points + 1):
            for j in range(-grid_points+1, grid_points+1, 1):
                for i in range(-grid_points+1, grid_points+1, 1):
                    W = (B
                     + C[:,:,0] * np.exp(1j*k*stepsize) * np.exp(-factor*si*tau[1])
                     + C[:,:,1] * np.exp(1j*j*stepsize) * np.exp(-factor*si*tau[2])
                     + C[:,:,2] * np.exp(1j*j*stepsize) * np.exp(-factor*si*tau[3]))
                    r = linalg.eig(W, E, left=False, right=False)
                    gk.append(np.conjugate(r))
    else:
        raise NotImplementedError(f"Not implemented for more than 3 non-zero delays.")
    return np.concatenate(gk)
    
def compute_n_rhp(E, B, C, tau: npt.NDArray, basic_delay: float=None, **kwargs) -> int:
    """

    Args:
        basic_delay (float):default None means delays are not commensurate

    kwargs:
        n_minimal (int): minimal degree of discretization, default 8
        n_grid (int): number of grids point in interval [0, pi] for discretizing
            \Psi, default 20
    
    Returns:
        n (int): number of discretization points necessary

    See:
     [1] Wu, Z., & Michiels, W. Reliably computing all characteristic roots of
         delay differential equations in a given right half plane using a 
         spectral method. Journal of Computational and Applied Mathematics,
         236(9), 2012, pp. 2499-2514.
    """

    # TODO checks

    N_minimal = kwargs.get("n_minimal", 8)
    n_grid = kwargs.get("n_grid", 20) # TODO assert ge 0
    mA = len(tau) # number of delays

    is_commmensurate = False
    if basic_delay is not None:
        # check that delays are indeed commensurate
        n_k = tau / basic_delay
        if np.any(np.abs(n_k - np.round(n_k)) >= 1e-10):
            raise ValueError(f"The provided delays are not commensurate with {basic_delay=}")
        n_k = n_k.astype(dtype=int)
        is_commmensurate = True
    elif mA > 4:
        logger.debug(f"More thatn three delays, delays will be approximated by commensurate delays with basic_delay=1/20")
        NN = 20
        n_k = np.round(tau*NN).astype(dtype=int)
        is_commmensurate = True
    else:
        pass # TODO maybe some log message

    #h=np.pi / n_grid # step size
    factor = 1.05*np.sin(np.pi/n_grid)

    if is_commmensurate:
        gk = commensurate_gk(E, B, C, n_k, grid_points=n_grid)
    else:
        # cases for 2, 3, 4 delays
        gk = disproportionate_gk(E, B, C, tau, grid_points=n_grid)

    #gk = np.concatenate(gk)
    gk = gk[np.isfinite(gk)] # get rid of inf and NaN
    si = np.max(np.real(gk))

    if si <= 0:
        # TODO warning, as per original Pieter comment:
        #    "What does this mean? Should we give a warning? When can this happen?"
        return N_minimal
    else:
        # matlab code - line 140 -> TODO to function (repeating code)
        points = gk[(np.real(gk) >= 0) & (np.real(gk) <= factor * si)]
        if np.size(points) == 0: # points are empty
            N1 = 0
        else:
            theta_points = np.abs(np.angle(points))
            pp_a = cubic_spline_a(theta_points)
            pp_b = cubic_spline_b(theta_points)
            r_points = np.abs(points)
            N_gk = (r_points - pp_b) / pp_a
            N1 = np.max(N_gk)

        # matlab code - line 157 to 200
        if is_commmensurate:
            # commensurate delays
            gk = commensurate_gk2(E, B, C, tau, n_k, si, grid_points=n_grid)
        else:
            # cases for 2, 3, 4 delays
            gk = disproportionate_gk2(E, B, C, tau, si, grid_points=n_grid)
            print(gk)
        
        # matlab code - line 202 -> TODO to function (repeating code)
        gk = gk[np.isfinite(gk)] # get rid of inf and NaN
        print(factor * si)
        points = gk[np.real(gk) >= factor * si]
        if np.size(points) == 0: # points are empty
            N2 = 0
        else:
            theta_points = np.abs(np.angle(points))
            pp_a = cubic_spline_a(theta_points)
            pp_b = cubic_spline_b(theta_points)
            r_points = np.abs(points)
            N_gk = (r_points - pp_b) / pp_a
            N2 = np.max(N_gk)
        
        print(f"{N1=} {N2=} {N_minimal=}")
        return max(np.ceil(N1), np.ceil(N2), N_minimal)






if __name__ == "__main__":
    # test case
    basic_delay = 1.5
    E = np.array([[1,0,0], [0,1,0], [0,0,0]], dtype=np.float64)
    B = np.array([[0.3116072141901096, 0.40602145612523965, 0.6776544796580569],
 [0.11363668625750645, 0.9343699659543538, 0.6977236310729171],
 [0.2941822873723423, 0.4997871977470306, 0.6949324821183755]], dtype=np.float64)
    C = np.array([[[0.7622980349226127, 0.22478751290330823],
  [0.48683084185446224, 0.8488352460039917],
  [0.32647364799957856, 0.02564225148608379]],
 [[0.3306919345179705, 0.4461952281462491],
  [0.5706006218521886, 0.17014364199309917],
  [0.4775272358483136, 0.9953991667479567]],
 [[0.44686477247542555, 0.17775612836209154],
  [0.008192538279984674, 0.5716834321644337],
  [0.6357900350689626, 0.03723974619198067]]], dtype=np.float64)
    tau = np.array([0, 0.05, 0.078])

    n = compute_n_rhp(E, B, C, tau)
    print(f"Discretization = {n}")