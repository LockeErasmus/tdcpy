# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

"""
Discretization heuristic
------------------------

References:

 [1] Wu, Z., & Michiels, W. Reliably computing all characteristic roots of
     delay differential equations in a given right half plane using a 
     spectral method. Journal of Computational and Applied Mathematics,
     236(9), 2012, pp. 2499-2514.

"""

import itertools
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
THETA: npt.NDArray = np.linspace(0, np.pi/2, 33, dtype=np.float64)

B_THETA_RECT = np.array([-0.2462, -0.2266, -0.2053, -0.2302, -0.2326, -0.2335,
                         -0.2362, -0.2421, -0.2463, -0.2604, -0.2656, -0.2749,
                         -0.2919, -0.3030, -0.3140, -0.3265, -0.3491, -0.3664,
                         -0.3892, -0.4204, -0.4548, -0.4855, -0.5339, -0.5872,
                         -0.6491, -0.7354, -0.8478, -0.9930, -1.1800, -1.4448,
                         -1.9414, -2.7149, -3.0292, -2.7272, -2.2329, -1.9425,
                         -1.7025, -1.5274, -1.3828, -1.2578, -1.1550, -1.0718,
                         -0.9977, -0.9340, -0.8771, -0.8243, -0.7852, -0.7499,
                         -0.7131, -0.6879, -0.6574, -0.6386, -0.6108, -0.5970,
                         -0.5816, -0.5640, -0.5540, -0.5413, -0.5341, -0.5273,
                         -0.5187, -0.5147, -0.5130, -0.5133, -0.5123])
A_THETA_RECT = np.array([0.9124, 0.9123, 0.9136, 0.9165, 0.9195, 0.9234, 0.9285,
                         0.9345, 0.9416, 0.9501, 0.9592, 0.9698, 0.9818, 0.9947,
                         1.0090, 1.0249, 1.0427, 1.0620, 1.0833, 1.1069, 1.1331,
                         1.1614, 1.1936, 1.2289, 1.2685, 1.3132, 1.3642, 1.4231,
                         1.4913, 1.5731, 1.6783, 1.7867, 1.8183, 1.7507, 1.6451,
                         1.5566, 1.4801, 1.4158, 1.3602, 1.3108, 1.2670, 1.2281,
                         1.1935, 1.1621, 1.1333, 1.1073, 1.0841, 1.0630, 1.0433,
                         1.0260, 1.0098, 0.9958, 0.9823, 0.9709, 0.9603, 0.9508,
                         0.9427, 0.9356, 0.9295, 0.9246, 0.9202, 0.9170, 0.9148,
                         0.9136, 0.9133])
THETA_RECT= np.linspace(0, np.pi, 65)

cubic_spline_a = interpolate.CubicSpline(THETA, A_THETA)
cubic_spline_b = interpolate.CubicSpline(THETA, B_THETA)

cubic_spline_a_rect = interpolate.CubicSpline(THETA_RECT, A_THETA_RECT)
cubic_spline_b_rect = interpolate.CubicSpline(THETA_RECT, B_THETA_RECT)

def grid_generator(m: int, n_grid_points:int=10):
    """ Generator for m-dimensional space grid points in [0, pi] x [-pi, pi]^{m-1} 
    
    Parameters
    ----------
    m : int
        number of dimensions
    n_grid_points : int
        number of grid points in each dimension (default 10)

    Returns
    -------
    generator
        generator yielding m-dimensional grid points

    Notes
    -----
    1. first dimension is in [0, pi]
    2. other dimensions are in [-pi, pi]
    3. total number of points generated is n_grid_points**m

    Examples
    --------
    >>> for point in grid_generator(2, 3):
    ...     print(point)
    [0.         0.        ]
    [0.         3.14159265]
    [0.         -3.14159265]
    [1.57079633 0.        ]
    [1.57079633 3.14159265]
    [1.57079633 -3.14159265]
    [3.14159265 0.        ]
    [3.14159265 3.14159265]
    [3.14159265 -3.14159265]
    
    """
    if m < 1:
        raise ValueError("m must be >= 1")

    grid0 = np.linspace(0.0, np.pi, n_grid_points, endpoint=True)
    grid_rest = np.linspace(-np.pi, np.pi, n_grid_points, endpoint=True)
    grids = [grid0] + [grid_rest] * (m - 1)
    vec = np.empty(m, dtype=grid0.dtype)

    for sample in itertools.product(*grids):
        vec[:] = sample
        yield vec

def commensurate_gk(E, B, C, n_k, grid_points=20) -> npt.NDArray:
    """ TODO """
    gk = []
    jhh = np.pi / (grid_points*n_k[-1])
    for k in range(grid_points*n_k[-1]):
        W = B + np.sum(C * np.exp(1j*k*jhh*n_k[1:]), axis=2)
        r = linalg.eig(W, E, left=False, right=False)
        gk.append(np.conjugate(r)) # complex conjugate
    return np.concatenate(gk)

def incommensurate_gk(E, B, C, tau, grid_points=20) -> npt.NDArray:
    """ TODO - in original implementation, only 3 non zero delays allowed """
    n_delays = len(tau)
    if n_delays < 2:
        raise NotImplementedError("Imposible to calculate for 1 or 0 delays")
    elif n_delays < 5:
        gk = []
        for sample_vec in grid_generator(n_delays-1, grid_points + 1):
            W = B + np.sum(C * np.exp(1j * sample_vec), axis=2)
            r = linalg.eig(W, E, left=False, right=False)
            gk.append(np.conjugate(r))
        return np.concatenate(gk)
    else:
        # TODO, this implementation works for more delays, it raised error in original code
        raise NotImplementedError(f"Not implemented for more than 3 non-zero delays.") 
    
def compute_n_rhp(E, B, C, tau: npt.NDArray, basic_delay: float=None, **kwargs) -> int:
    """

    Args:
        basic_delay (float): define if delays are commensurate, default None

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

    is_commensurate = False
    if basic_delay is not None:
        # check that delays are indeed commensurate
        n_k = tau / basic_delay
        if np.any(np.abs(n_k - np.round(n_k)) >= 1e-10):
            raise ValueError(f"The provided delays are not commensurate with {basic_delay=}")
        n_k = n_k.astype(dtype=int)
        is_commensurate = True
    elif mA > 4:
        logger.debug(f"More than three delays, delays will be approximated by commensurate delays with basic_delay=1/20")
        NN = 20
        n_k = np.round(tau*NN).astype(dtype=int)
        is_commensurate = True
    else:
        pass # TODO maybe some log message

    #h=np.pi / n_grid # step size
    factor = 1.05*np.sin(np.pi/n_grid) # np.pi / n_grid <- stepsize

    if is_commensurate:
        gk = commensurate_gk(E, B, C, n_k, grid_points=n_grid)
    else: # cases for 2, 3, 4 delays
        gk = incommensurate_gk(E, B, C, tau, grid_points=n_grid)

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
        if is_commensurate:
            gk = commensurate_gk(E, B, C * np.exp(-factor * si * tau[1:]), n_k, grid_points=n_grid)
        else: # case for > 2 delays
            gk = incommensurate_gk(E, B, C * np.exp(-factor * si * tau[1:]), tau, grid_points=n_grid)
        
        # matlab code - line 202 -> TODO to function (repeating code)
        gk = gk[np.isfinite(gk)] # get rid of inf and NaN
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
        
        logger.debug(f"{N1=} {N2=} {N_minimal=}")
        return int(max(np.ceil(N1), np.ceil(N2), N_minimal))


def compute_n_rect(region: tuple[int], tau_max, **kwargs) -> int:
    """ Computes the deegree of the spectral discretization for rectanuglar
    region in complex plane

    Computes the degree of the spectral discretisation and a shift of the origin
    such that exp(-s*taum) is sufficiently well approximated in the rectanuglar
    region in complex plane:

    region[1]+1j*region[4] --------- region[2]+1j*region[4]
             |                                |
             |                                |
    region[1]+1j*region[3] --------- region[2]+1j*region[3]

    Args:
        region (tuple): tuple representing rectangular region in complex plane,
            example: (-10, 0, 0, 100) represents region D from complex plane
            D={z from C: -10<=Re(z)<=0, 0<=Im(z)<=100}
        tau_max (float): maximal delay, has to be >= 0
        kwargs:

    Returns:
        tuple containing

            - n (int): number of discretization points necessary
            - origin (complex): origin TODO
    
    See:
        [1] Wu, Z., & Michiels, W. (2012) Reliably computing all
            characteristic roots of delay differential equations in a given
            right half plane using a  spectral method. Journal of
            Computational and Applied Mathematics, 236(9), pp. 2499-2514.
    """

    n_minimal = 8 

    # unpack boundaries
    rmini, rmaxi, imini, imaxi = region

    # scaling by tau_max
    realmini = rmini * tau_max
    realmaxi = rmaxi * tau_max
    imagmini = imini * tau_max
    imagmaxi = imaxi * tau_max

    # create list of potential centers
    realpart = np.linspace(realmini, realmaxi, 101)
    origi = realpart + 1j * ((imagmini + imagmaxi) / 2.)

    # due to choice of potential origins, symmetric with respect the line
    # 1j*(imagmini+imagmaxi)/2 => we only need to consider the top vertices
    theta1_points = np.angle((realmaxi + 1j*imagmaxi) - origi) # angle (theta) (in [0,pi/2]) of the top right vertex for the different origins 
    pp1_b=cubic_spline_b_rect(theta1_points) # a(theta_top_right)
    pp1_a=cubic_spline_a_rect(theta1_points) # b(theta_top_right)
    r1_points = np.abs((realmaxi+1j*imagmaxi)-origi) # radius (r) of the top right vertex for the different origins 
    # N(theta_i) = (R_i - b(theta_i))/a(theta_i), number of descretization points necessary to accurately
    # approximate the top right vertex for the different origins
    n1 = (r1_points - pp1_b) / pp1_a

    theta2_points = np.angle((realmini + 1j*imagmaxi) - origi) # angle (theta) (in [pi/2,pi]) of the top left vertex for the different origins 
    pp2_b=cubic_spline_b_rect(theta2_points) # a(theta_top_left)
    pp2_a=cubic_spline_a_rect(theta2_points) # b(theta_top_left)
    r2_points=np.abs((realmini + 1j*imagmaxi) - origi) # radius (r) of the top left vertex for the different origins 
    # N(theta_i) = (R_i - b(theta_i))/a(theta_i), number of descretization points necessary to accurately
    # approximate the top left vertex for the different origins
    n2 = (r2_points - pp2_b) / pp2_a
    nn = np.maximum(n1, n2)
    n_index = np.argmin(nn)
    n = nn[n_index]
    n = n_minimal if n_minimal > n else n # make sure n >= n_minimal
    origin = origi[n_index]
    return int(n), origin

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