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
    
    Parameters
    ----------
    roots : npt.NDArray
        array of complex numbers to visualize
    ax : Axes
        matplotlib.axes.Axes object to plot to
    *args :
        see matplotlib .scatter function
    **kwargs :
        see matplotlib .scatter function    

    Returns
    -------
    None
    
    Notes
    -----
    1. if roots is empty, does nothing
    2. if ax is None, creates new figure and axes
    3. uses matplotlib scatter function internally
    4. x axis is real part, y axis is imaginary part

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from tdspy.plot.eigenvalues import complex_scatter_axplot
    >>> roots = np.array([1+2j, -1-1j, 0+0j, 3+0j])
    >>> fig, ax = plt.subplots()
    >>> complex_scatter_axplot(roots, ax, c='r', marker='x')
    >>> plt.show()
    """
    ax.scatter(np.real(roots), np.imag(roots), *args, **kwargs)


def eigen_plot(roots: npt.NDArray, ax=None, **kwargs):
    """ Plots eigenvalues in complex plane

    Parameters
    ----------
    roots : npt.NDArray
        array of complex eigenvalues to plot
    ax : Axes, optional
        matplotlib.axes.Axes object to plot to, default None
    **kwargs :
        tol (float): tolerance for assuming Re(root) ~ 0, default 1e-10

    Returns
    -------
    ax : Axes
        matplotlib.axes.Axes object containing the plot

    Notes
    -----
    1. if ax is None, creates new figure and axes
    2. plots horizontal and vertical lines at 0
    3. colors eigenvalues based on their real part:
        - red for Re(root) > tol
        - blue for |Re(root)| <= tol
        - green for Re(root) < -tol

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from tdspy.plot.eigenvalues import eigen_plot
    >>> roots = np.array([1+2j, -1-1j, 0+0j, 3+0j])
    >>> ax = eigen_plot(roots)
    >>> plt.show()
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


if __name__ == "__main__":
    import numpy as np
    import doctest
    doctest.testmod()