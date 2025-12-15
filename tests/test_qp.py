"""
Tests quasipolynomial representation
"""

import pytest

import numpy as np
import numpy.typing as npt

import tdspy
from tdspy.common.quasipoly import compress_qp, qp_to_ndde

def assert_arrays_equal(a: npt.NDArray, b: npt.NDArray):
    assert a.ndim == b.ndim, f"ndim mismatch: {a.ndim} != {b.ndim}"
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} != {b.shape}"
    assert np.array_equal(a, b), f"Arrays differ:\n{a}\nvs\n{b}"

@pytest.mark.parametrize(
        argnames="delays, coefs, ascending, ndde",
        argvalues=[
            (
                np.array([0, 1.]),
                np.array([[1, 0],[0, 0.]]),
                True,
                None,
            ),
            (
                np.array([0, 1.]),
                np.array([[0, 1],[1, 0.]]),
                True,
                None,
            ),
            (
                np.array([0, 1, 2.]),
                np.array([
                    [1, -2, 1],
                    [0, -2, 2],
                    [0, 0, 1.],
                ]),
                False,
                (# A, hA, H, hH
                    np.stack(
                        [
                            np.array([ # A0
                                [ 0., 1.],
                                [-1., 2.],
                            ]),
                            np.array([ # A1
                                [ 0.,  0.],
                                [-2.,  2.],
                            ]),
                            np.array([ # A2
                                [ 0., 0.],
                                [-1., 0.],
                            ]),
                        ],
                        axis=2
                    ),
                    np.array([0, 1, 2.]),
                    np.zeros(shape=(2,2,0)),
                    np.array([], dtype=np.float64),
                )
            ),
            (
                np.array([0, 1, 1.5]),
                np.array([[3.,1.5],[2.0,0.5],[-2.0,-0.5]]),
                True,
                (# A, hA, H, hH
                    np.stack(
                        [
                            np.array([ # A0
                                [-2.],
                            ]),
                            np.array([ # A1
                                [-4/3.],
                            ]),
                            np.array([ # A2
                                [ 4/3.],
                            ]),
                        ],
                        axis=2
                    ),
                    np.array([0, 1, 1.5]),
                    np.stack(
                        [
                            np.array([ # H1
                                [1/3.],
                            ]),
                            np.array([ # H2
                                [-1/3.],
                            ]),
                        ],
                        axis=2
                    ),
                    np.array([1, 1.5]),
                )
            ),
        ],
        ids=[
            "simple-01",
            "simple-02",
            "MATLAB-tds-control-retarded",
            "MATLAB-tds-control-neutral",
        ],
)
def test_qp_to_ndde(coefs: npt.NDArray, delays: npt.NDArray, ascending: bool, ndde: tuple | None):
    """ Tests quasipolynomial -> NDDE  convert """
    A, hA, H, hH = qp_to_ndde(coefs, delays, ascending=ascending)

    if ndde is not None:
        # unpack expected solution
        expected_A, expected_hA, expected_H, expected_hH = ndde

        assert_arrays_equal(A, expected_A)
        assert_arrays_equal(hA, expected_hA)
        assert_arrays_equal(H, expected_H)
        assert_arrays_equal(hH, expected_hH)