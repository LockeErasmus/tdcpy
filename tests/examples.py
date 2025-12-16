"""
Docstring for test.examples
"""

import numpy as np
import numpy.typing as npt


def tds_control_manual_example_2_6(tau2=2.0) -> npt.NDArray[np.float64]:
    """ Example 2.1 from TDS Control Manual
    Generates the system matrices for the example in the TDS Control Manual, Section 2.1.
    Returns:
        A numpy array representing the system matrices.
    """
    A = np.stack([
        np.array([[0.25]]),
        np.array([[-1./3]]),
    ], axis=2)
    hA = np.array([0, 1.])
    H = np.stack([
        np.array([[-0.75]]),
        np.array([[0.5]]),
    ], axis=2)
    hH = np.array([1., tau2])

    return A, hA, H, hH


import pytest

TDS_CONTROL_MANUAL_NEUTRAL_CASES = [
    pytest.param(
        tds_control_manual_example_2_6(tau2=2.0),
        id="tds_control_manual_example_2_6_tau2=2.0",
    ),
    pytest.param(
        tds_control_manual_example_2_6(tau2=2.05),
        id="tds_control_manual_example_2_6_tau2=2.05",
    ),
]