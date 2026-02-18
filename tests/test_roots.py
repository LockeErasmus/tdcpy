# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

"""
Tests for tdcpy.roots
"""

import logging
from imagesize import get
import pytest

import numpy as np
import tdcpy

import tests.examples

CASES = [
    pytest.param(
        tests.examples.tds_control_manual_example_2_1(),
        { # options
            "args": (),
            "kwargs": {"r": -2.5},
            "expected_results": None,
        },
        id="tds_control_manual_example_2_1",
    ),
]

@pytest.mark.parametrize(
    argnames="rdde, options",
    argvalues=CASES,
)
def test_roots_rdde(rdde, options, enable_plot: bool) -> None:    
    # unpack test case
    A, hA = rdde    
    
    args = options["args"]
    kwargs = options["kwargs"]
    expected_results = options["expected_results"]

    # create RDDE
    import tdcpy
    rdde = tdcpy.RDDE(A=A, hA=hA)
    cr, info = tdcpy.roots(rdde, *args, **kwargs)

    if expected_results is not None:
        raise NotImplementedError("Test case has expected results defined, but assertion is not implemented yet.")

    if enable_plot:
        import matplotlib.pyplot as plt
        import tdcpy.plot
        
        fig, axes = plt.subplots(2, 3)
        tdcpy.plot.eigen_plot(cr, ax=axes[0, 0], title="Roots of RDDE", xlabel="Real part", ylabel="Imaginary part")

        # count roots
        n = len(cr)

        # discretization plots
        minRe = np.min(np.real(info.discretization_eigenvalues))
        maxRe = np.max(np.real(info.discretization_eigenvalues))
        
        tdcpy.plot.complex_scatter_axplot(info.discretization_eigenvalues, ax=axes[1, 0], marker="o", color="b",
                                          facecolor="none")
        idx = np.argsort(np.real(info.discretization_eigenvalues))[::-1]
        nd = len(info.discretization_eigenvalues)
        cr1 = info.discretization_eigenvalues[idx][:min((n + nd)//2, nd) - 1]
        cr2 = info.discretization_eigenvalues[idx][:min(2*n, nd) - 1]
        tdcpy.plot.complex_scatter_axplot(cr1, ax=axes[1, 1], marker="o", color="b", facecolor="none")
        tdcpy.plot.complex_scatter_axplot(cr2, ax=axes[1, 2], marker="o", color="b", facecolor="none")

        
        
        plt.show()