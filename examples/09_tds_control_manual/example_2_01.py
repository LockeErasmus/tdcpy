# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

r"""
Example 2.1 - Stability analysis of retarded DDE
================================================

We consider the following retarded delay differential equation (RDDE) from
:cite:`verheyden2008efficient` Section 6.1:

.. math::

    \dot{x}(t) =
        \begin{bmatrix}
            -1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & -10 & -4 \\
            0 & 0 & 4 & -10
        \end{bmatrix} x(t)
        + 
        \begin{bmatrix}
            3 & 3 & 3 & 3\\
            0 & -1.5 & 0 & 0 \\
            0 & 0 & 3 & -5 \\
            0 & 5 & 5 & 5
        \end{bmatrix} x(t-1).

We will follow the steps from :cite:`appeltans2023analysis` Section 2.2 to
achive the same results as presented there, i.e. we will

1. create the `RDDE` matrix representation
2. compute characteristic roots via the `tdcpy.roots` function
3. plot the computed characteristic roots using `tdcpy.plot.eigen_plot` function

.. bibliography::
    :filter: {"auto_examples/tds_control_manual/example_2_1"} & docnames

"""

import matplotlib.pyplot as plt
import numpy as np
import tdcpy
import tdcpy.plot

# Create RDDE matrix representation
A0 = np.array([[-1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, -10, -4],
               [0, 0, 4, -10]])
A1 = np.array([[3, 3, 3, 3],
               [0, -1.5, 0, 0],
               [0, 0, 3, -5],
               [0, 5, 5, 5]])

rdde = tdcpy.RDDE(A=[A0, A1], hA=[0, 1])
cr, info = tdcpy.roots(rdde, r=-2.5)

tdcpy.plot.eigen_plot(cr)
plt.show()

