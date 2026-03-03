"""
Notorious examples of functions I can use for testing quazi-newton algorithms.
"""

import numpy as np

def rosenbrock_2d(x, a=1, b=100):
    x = np.asarray(x, dtype=float)
    if len(x) != 2:
        raise ValueError("Length of x must be equal to 2")
    fx = (a - x[0])**2 + b * (x[1] - x[0]**2) ** 2
    dfx = np.array([2 * (x[0] - 1) - 4 * b * (x[1] - x[0]**2)*x[0],
                    2 * b * (x[1] - x[0]**2)])
    return fx, dfx

def rosenbrock_nd(x, n, a=1, b=100):
    x = np.asarray(x, dtype=float)
    if len(x) != n:
        raise ValueError("Length of x must be equal to n")

    fx = 0.0
    for i in range(n - 1):
        fx += (a - x[i])**2 + b * (x[i+1] - x[i]**2)**2

    dfx = np.zeros(n)
    for i in range(n):
        if i == 0:
            dfx[i] = (
                -2*(a - x[i])
                - 4*b*x[i]*(x[i+1] - x[i]**2)
            )
        elif i == n - 1:
            dfx[i] = 2*b*(x[i] - x[i-1]**2)
        else:
            dfx[i] = (
                -2*(a - x[i])
                - 4*b*x[i]*(x[i+1] - x[i]**2)
                + 2*b*(x[i] - x[i-1]**2)
            )

    return fx, dfx

def powell_singular(x, n):
    x = np.asarray(x, dtype=float)
    if len(x) != n:
        raise ValueError("Length of x must equal n")
    if n % 4 != 0:
        raise ValueError("n must be divisible by 4")

    fx = 0.0
    dfx = np.zeros(n)

    for k in range(0, n, 4):
        x1 = x[k]
        x2 = x[k+1]
        x3 = x[k+2]
        x4 = x[k+3]

        t1 = x1 + 10*x2
        t2 = x3 - x4
        t3 = x2 - 2*x3
        t4 = x1 - x4

        fx += t1**2 + 5*t2**2 + t3**4 + 10*t4**4

        dfx[k]   += 2*t1 + 40*t4**3
        dfx[k+1] += 20*t1 + 4*t3**3
        dfx[k+2] += 10*t2 - 8*t3**3
        dfx[k+3] += -10*t2 - 40*t4**3

    return fx, dfx

def abs_rosenbrock_nd(x, n, a=1, b=100):
    x = np.asarray(x, dtype=float)
    if len(x) != n:
        raise ValueError("Length of x must equal n")
    fx = 0.0
    dfx = np.zeros(n)

    for i in range(n-1):
        t1 = a - x[i]
        t2 = x[i+1] - x[i]**2
        fx += t1**2 + b * abs(t2)

    for i in range(n):
        grad = 0.0
        if i > 0:
            t_prev = x[i] - x[i-1]**2
            grad += b * np.sign(t_prev)
        if i < n-1:
            t_next = x[i+1] - x[i]**2
            grad += -2*b*x[i]*np.sign(t_next)
        grad += -2*(a - x[i])
        dfx[i] = grad
    
    return fx, dfx