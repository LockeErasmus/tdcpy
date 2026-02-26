# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl

from tdcpy.optimize.bfgs import bfgs_weak_wolfe
import scipy.optimize
import numpy as np

from .examples import rosenbrock_2d, rosenbrock_nd, powell_singular, abs_rosenbrock_nd

def test_rosenbrock():

    x0 = np.array([3.5, 20.])
    # opt = #[a, a**2]

    sol = scipy.optimize.minimize(
        rosenbrock_2d, x0=x0, jac=True, args=(1, 100),
        method=bfgs_weak_wolfe, options={"disp": True}
    )

    print(sol.x)

def test_rosenbrock2():

    np.random.seed(42)
    x0 = 100*np.random.rand(20)

    sol = scipy.optimize.minimize(
        rosenbrock_nd, x0=x0, jac=True, args=(20, 1, 100),
        method=bfgs_weak_wolfe, options={"disp": True, "maxiter": 2000}
    )

    print(sol.x)

def test_powell():

    np.random.seed(42)
    x0 = 100*np.random.rand(20)

    sol = scipy.optimize.minimize(
        powell_singular, x0=x0, jac=True, args=(20,),
        method=bfgs_weak_wolfe, options={"disp": True, "maxiter": 2000}
    )

    print(sol.x)

def test_abs_rosenbrock_nd():
    np.random.seed(42)
    x0 = 100*np.random.rand(20)

    sol = scipy.optimize.minimize(
        abs_rosenbrock_nd, x0=x0, jac=True, args=(20, 1, 100),
        method=bfgs_weak_wolfe, options={"disp": True, "maxiter": 2000}
    )

    print(sol.x)



