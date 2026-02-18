# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

"""
Docstring for tdcpy.utils
"""

import numpy as np
import numpy.typing as npt

from .ddae import DDAE

def validate_rectangular_region(r: list|tuple|npt.NDArray ) -> tuple[float, float, float, float]:
    """ Validates the rectangular region definition and returns the coordinates
    as tuple
    Args:
        r (list|tuple|array): rectangular region defined by 4 coordinates
        [a, b, c, d] where a < b and c < d    
    """
    if not isinstance(r, (list, tuple, np.ndarray)) or len(r) != 4:
        raise ValueError(("Rectangular region has to be defined as list, tuple or array "
                          "with 4 elements [a, b, c, d] where a < b and c < d"))

    a, b, c, d = r
    if not (a < b and c < d):
        raise ValueError(("Rectangular region has to be defined by 4 coordinates [a, b, c, d] where a < b and c < d"))

    return float(a), float(b), float(c), float(d)


def print_system_matrices(ddae: DDAE, **kwargs):
    with np.printoptions(precision=4, linewidth=1000, suppress=True):
        print(f"E")
        print(ddae.E)
        print("-"*50)
        for i in range(ddae.mA):
            print(f"A[:,:,{i} - tau={ddae.hA[i]}")
            print(ddae.A[:,:,i])
            print("-"*50)
        for i in range(ddae.mB):
            print(f"B[:,:,{i} - tau={ddae.hB[i]}")
            print(ddae.B[:,:,i])
            print("-"*50)
        for i in range(ddae.mC):
            print(f"C[:,:,{i} - tau={ddae.hC[i]}")
            print(ddae.C[:,:,i])
            print("-"*50)
        for i in range(ddae.mD):
            print(f"D[:,:,{i} - tau={ddae.hD[i]}")
            print(ddae.D[:,:,i])
            print("-"*50)