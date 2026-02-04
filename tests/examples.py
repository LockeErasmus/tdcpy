"""
Docstring for test.examples
"""

import numpy as np
import numpy.typing as npt

def tds_control_manual_example_2_1() -> npt.NDArray[np.float64]:
    # Create RDDE matrix representation
    A0 = np.array([[-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -10, -4],
                [0, 0, 4, -10]])
    A1 = np.array([[3, 3, 3, 3],
                [0, -1.5, 0, 0],
                [0, 0, 3, -5],
                [0, 5, 5, 5]])
    return np.stack([A0, A1], axis=2), np.array([0, 1.0])

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

TDS_CONTROL_RETARDED_CASES = []

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



STABOPT_NEUTRAL_EXAMPLES = [
    pytest.param(
        (
            A0 = np.array([
        [   -0.08,  -0.03,  0.2     ],
        [   0.2,    -0.04,  -0.005  ],
        [   -0.06,  0.2,    -0.07    ],
    ])
    A = np.stack([A0], axis=2)
    hA = np.array([0])
    Bu = np.array([ [-0.1],[-0.2],[0.1]    ])
    B = np.stack([Bu],axis=2)
    hB = np.array([5.])
    C = np.array(np.eye(3))
    C = np.stack([C],axis=2)
    hC = np.array([0])
    D1 = np.array([ [3],[4],[1]    ])
    D2 = np.array([ [0.4],[-0.4],[-0.4]])
    D = np.stack([D1,D2],axis = 2)
    hD = np.array([2.5,5.])
        )
        id="01" # TODO
    ),
]