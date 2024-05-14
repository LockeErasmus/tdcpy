"""

Original matlab files:
    - tds-control/code/+tds_control/+stability/lowerbound.m
    - tds-control/code/+tds_control/+stability/upperbound.m
"""

def lower_bound(x: float, epsilon: float=0.15, gamma: float=1e-6):
    """ Calculates lower bound """
    if x <= -gamma/epsilon:
        return x*(1. + epsilon)
    elif x >= gamma/epsilon:
        return x*(1. - epsilon)
    else:
        return x - gamma

def upper_bound(x: float, epsilon: float=1e-2, gamma: float=1e-2):
    """ Calculates upper bound """
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


