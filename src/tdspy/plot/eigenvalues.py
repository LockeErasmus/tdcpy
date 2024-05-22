"""
Set of ploting eigenvalue plotting functions
--------------------------------------------

Mainly to provide similar functionality like the original TDS-CONTROl function:

https://gitlab.kuleuven.be/u0011378/tds-control/-/blob/main/tds-control/code/tds_eigenplot.m
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def eigen_plot(roots1, roots0=None, ax=None, **kwargs):
    """
    
    Args:
        tol (float): tolerance for assuming Re(root) ~ 0, defgault 1e-10
    
    """

    tol = kwargs.get("tol", 1e-10)

    ax_was_none = False
    if ax is None:
        ax_was_none = True
        fig, ax = plt.subplots()
    else:
        raise NotImplementedError("...")
    
    ax.axhline(0.0, linestyle="-.", linewidth=1, color="k")
    ax.axvline(0.0, linestyle="-.", linewidth=1, color="k")
    
    roots1_real = np.real(roots1)
    roots1_imag = np.imag(roots1)

    mask_negative = roots1_real < -tol
    if np.any(mask_negative) > 0:
        ax.scatter(roots1_real[mask_negative],
                   roots1_imag[mask_negative],
                   marker="x",
                   color="g",
                   linewidths=0.5,
        )
    
    mask_positive = roots1_real > tol
    if np.any(mask_positive) > 0:
        ax.scatter(roots1_real[mask_positive],
                   roots1_imag[mask_positive],
                   marker="x",
                   color="r",
                   linewidths=0.5,
        )

    mask_zero = ~(mask_positive | mask_negative)
    if np.any(mask_zero) > 0:
        ax.scatter(roots1_real[mask_zero],
                   roots1_imag[mask_zero],
                   marker="x",
                   color="b",
                   linewidths=0.5,
        )
    
    ax.set_xlabel(r"$\Re (\lambda)$")
    ax.set_ylabel(r"$\Im (\lambda)$")
    
    return ax



