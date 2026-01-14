"""
Test discretization heuristic
"""
import numpy as np
import pytest

from tdspy.stability.discretization_heuristic import compute_n_rhp

A0 = np.array([[-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -10, -4],
                [0, 0, 4, -10]])
A1 = np.array([[3, 3, 3, 3],
            [0, -1.5, 0, 0],
            [0, 0, 3, -5],
            [0, 5, 5, 5]])

CASES = [
    pytest.param(
        (np.eye(4), np.stack([A0, A1], axis=2), np.array([0, 1.0])),
        id="tds_control_manual_example_2_1",
    )
]



@pytest.mark.parametrize(
    argnames="factory",
    argvalues=CASES,
)
def test_discretization_heuristic(factory, enable_plot: bool) -> None:
    """ ... """
    E, A, hA = factory

    D, hD = ndde_to_diff(H, hH)
    # normalization is not necessary for NDDE
    print(D, hD)

    val, info = gamma_normalized_diff(H, hH, r=0, n_theta=3)   
    
    print(val, info)

    if enable_plot:
        raise NotImplementedError("Plotting not implemented yet for gamma_r tests.")