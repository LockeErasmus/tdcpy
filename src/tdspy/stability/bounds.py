# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

"""
Set of functions to compute the lower bounds and upper bounds
-------------------------------------------------------------

Implemented functions:
    1. lower_bound -> returns the lower bound at x, given (x,epsilon,gamma)
    2. upper_bound -> returns the upper bound at x, given (x,epsilon,gamma)

Original matlab files:
    - tds-control/code/+tds_control/+stability/lowerbound.m
    - tds-control/code/+tds_control/+stability/upperbound.m
"""

def lower_bound(x: float, epsilon: float=0.15, gamma: float=1e-6):
    """ Calculates lower bound 
    
    Parameters
    ----------
    x : float
        point to calculate lower bound at
    epsilon : float
        small positive parameter, default 1e-2
    gamma : float
        small positive parameter, default 1e-2 

    Returns
    -------
    float
        lower bound at x

    Notes
    -----
    1. for |x| large enough, lower bound is (1-epsilon)*x or
       (1+epsilon)*x
    2. for |x| small enough, lower bound is x - gamma

    Examples
    --------
    >>> from tdspy.stability.bounds import lower_bound, upper_bound
    >>> lower_bound(100.0)
    85.0
    >>> lower_bound(-100.0)
    -115.0
    >>> lower_bound(0.0)
    -1e-06
    >>> upper_bound(100.0)
    115.0
    >>> upper_bound(-100.0)
    -85.0
    >>> upper_bound(0.0)
    1e-06
    """
    if x <= -gamma/epsilon:
        return x*(1. + epsilon)
    elif x >= gamma/epsilon:
        return x*(1. - epsilon)
    else:
        return x - gamma

def upper_bound(x: float, epsilon: float=1e-2, gamma: float=1e-2):
    """ Calculates upper bound 
    
    Parameters
    ----------
    x : float
        point to calculate upper bound at
    epsilon : float
        small positive parameter, default 1e-2
    gamma : float
        small positive parameter, default 1e-2
    
    Returns
    -------
    float
        upper bound at x

    Notes
    -----
    1. for |x| large enough, upper bound is (1+epsilon)*x or
       (1-epsilon)*x
    2. for |x| small enough, upper bound is x + gamma

    Examples
    --------
    >>> from tdspy.stability.bounds import lower_bound, upper_bound
    >>> lower_bound(100.0)
    85.0
    >>> lower_bound(-100.0)
    -115.0
    >>> lower_bound(0.0)
    -1e-06
    >>> upper_bound(100.0)
    115.0
    >>> upper_bound(-100.0)
    -85.0
    >>> upper_bound(0.0)
    1e-06
    """
    if x <= -gamma/epsilon:
        return x*(1. - epsilon)
    elif x>= gamma/epsilon:
        return x*(1. + epsilon)
    else:
      return x+gamma

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    epsilon = 0.1
    gamma = 0.1
    x = np.linspace(-5*gamma/epsilon, 5*gamma/epsilon, 100)
    lb = np.vectorize(lower_bound)(x, epsilon, gamma)
    ub = np.vectorize(upper_bound)(x, epsilon, gamma)

    plt.figure()
    plt.plot(x, x, alpha=0.5, label="f(x)=x")
    plt.axvline(gamma/epsilon, color="r", alpha=0.5, label="+gamma/epsilon")
    plt.axvline(-gamma/epsilon, color="r", alpha=0.5, label="-gamma/epsilon")
    plt.plot(x, lb, label="lower bound")
    plt.plot(x, ub, label="lower bound")
    plt.legend()
    plt.show()


