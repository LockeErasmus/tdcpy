"""
Set of functions for animation of discretization
------------------------------------------------

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.linalg as linalg
from tdspy.common.discretization import discretize_ddae
from tdspy.rdde import RDDE
from tdspy.ndde import NDDE
from tdspy.ddae import DDAE

import logging

logger = logging.getLogger(__name__)

def discretization_animation(tds: RDDE | NDDE | DDAE, discretization, s0: complex=0j, discretization_ideal=None, xlim=None, ylim=None) -> animation.FuncAnimation:
    """ Creates discretization animation

    Args:
        discretization (iterable): see documentation for matplotlib.animation. FuncAnimation
            also, have in mind, that minimal discretization=8
        s0 (int): point discretization is done around, default 0
        discretization_ideal (int): this discretization is assumed to be
            'correct' and always present in animation, set None to turn off,
            default None
        xlim (tuple): x axis limits, default None
        ylim (tuple): y axis limits, default None
    
    Returns:
        ani (Animation): matlab Animation object, use plt.show() to see

    Notes:
        1. if this function does not suite to you, copy and rewrite this
        1. if you want to save animation
    """
    fig, ax = plt.subplots()

    if isinstance(tds, NDDE):
        tds = tds.to_ddae()
    
    # unpack into matrices
    E = tds.E
    A = tds.A
    hA = tds.hA

    if discretization_ideal is not None:
        # discretize and solve EVP
        Pi_N, Sigma_N = discretize_ddae(E, A, hA, discretization=discretization_ideal, s0=s0)
        raw_roots = linalg.eig(Sigma_N, Pi_N, left=False, right=False)
        raw_roots = raw_roots[np.isfinite(raw_roots)] # get rid of inf and NaN

        ax.scatter(np.real(raw_roots), np.imag(raw_roots), marker="x",
                    color="b", label=f"Solution of EVP N={discretization_ideal}")
    
    s = ax.scatter([], [], marker="o", edgecolors="r", facecolors='none', label="Solution of EVP N=")
    ax.set_xlabel(r"$\Re (\lambda)$")
    ax.set_ylabel(r"$\Im (\lambda)$")
    legend = ax.legend()

    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    def update(n):
        # discretize and solve EVP
        Pi_N, Sigma_N = discretize_ddae(E, A, hA, discretization=n, s0=s0)
        raw_roots = linalg.eig(Sigma_N, Pi_N, left=False, right=False)
        raw_roots = raw_roots[np.isfinite(raw_roots)] # get rid of inf and NaN

        s.set_offsets(np.column_stack([np.real(raw_roots), np.imag(raw_roots)]))
        legend.get_texts()[1].set_text("Solution of EVP N={}".format(n))
    
    ani = animation.FuncAnimation(fig, update, frames=discretization, interval=200)

    return ani