"""
Set of ploting eigenvalue plotting functions
--------------------------------------------

Mainly to provide similar functionality like the original TDS-CONTROl function:

https://gitlab.kuleuven.be/u0011378/tds-control/-/blob/main/tds-control/code/tds_eigenplot.m
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import numpy.typing as npt


def complex_scatter_axplot(roots: npt.NDArray, ax: Axes, *args, **kwargs) -> None:
    """ Plots non-empty complex numbers, x: Re(z), y: Im(z)
    
    Args:
        roots (array) array of complex numbers to visualize
        ax (Axes): matplotlib.axes.Axes object to plot to
        *args: see matplotlib .scatter function
        *kwargs: see matplotlib .scatter function
    """
    ax.scatter(np.real(roots), np.imag(roots), *args, **kwargs)


def eigen_plot(roots: npt.NDArray, ax=None, **kwargs):
    """
    
    Args:
        tol (float): tolerance for assuming Re(root) ~ 0, defgault 1e-10
    
    """

    tol = kwargs.get("tol", 1e-10)

    ax_was_none = False
    if ax is None:
        ax_was_none = True
        fig, ax = plt.subplots()
    
    ax.axhline(0.0, linestyle="-.", linewidth=1, color="k")
    ax.axvline(0.0, linestyle="-.", linewidth=1, color="k")
    
    roots_real = np.real(roots)

    conditions = [
        roots_real < -tol,
        (roots_real >= -tol) & (roots_real <= tol),
        roots_real > tol,
    ]
    colors = np.select(conditions, ["g", "b", "r"], default='gray')

    complex_scatter_axplot(roots, ax=ax, c=colors, marker="x", linewidth=0.5, label="roots")
    
    ax.set_xlabel(r"$\Re (\lambda)$")
    ax.set_ylabel(r"$\Im (\lambda)$")
    
    return ax



