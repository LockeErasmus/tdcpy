# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

"""
Docstring for test.stability.test_gamma_r
"""

import logging
import pytest

import tests.examples

import numpy as np
from tdcpy.stability.characteristic_roots import rightmost_root

def problem_20260220():
    """ Problem specifically connected to RMR obtained in BFGS optimization """
    E = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])

    A0 = np.array([
        [-0.08, -0.03, 0.2, 0, 0, 0, 0],
        [0.2, -0.04, -0.005, 0, 0, 0, 0],
        [-0.06, 0.2, -0.07, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, -1],
        [0, 1, 0, 0, 0, -1, 0],
        [0, 0, 1, 0, -1, 0, 0],
        [0, 0, 0, -1, 0, 0, 0]
    ])

    A1 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 3, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])

    A2 = np.array([
        [0, 0, 0, -0.1, 0, 0, 0],
        [0, 0, 0, -0.2, 0, 0, 0],
        [0, 0, 0, 0.1, 0, 0, 0],
        [0, 0, 0, 0.4, 0, 0, 0],
        [0, 0, 0, -0.4, 0, 0, 0],
        [0, 0, 0, -0.4, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ])

    A3 = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1.146471826538495, -0.485057279586679, 0.235365617136169]
    ])

    hA = np.array([0, 2.5, 5, 0])

    return E, np.stack([A0, A1, A2, A3], axis=2), hA
    

CASES = [
    pytest.param(
        (np.eye(4), ) + tests.examples.tds_control_manual_example_2_1(),
        0,
        {},
        {"rmr": 0.6176424667764324 + 0j},
        id="tds_control_manual_example_2_1_r=0",
    ),
    pytest.param(
        (np.eye(4), ) + tests.examples.tds_control_manual_example_2_1(),
        -1,
        {},
        {"rmr": 0.6176424667764324 + 0j},
        id="tds_control_manual_example_2_1_r=-1",
    ),
    pytest.param(
        (np.eye(4), ) + tests.examples.tds_control_manual_example_2_1(),
        -10,
        {},
        {"rmr": 0.6176424667764324 + 0j},
        id="tds_control_manual_example_2_1_r=-10",
    ),
    pytest.param(
        problem_20260220(),
        0,
        {},
        {"rmr": -0.001118911788916191-0.0011739231139356607j},
        id="problem-20260220_r=0",
    ),
    pytest.param(
        problem_20260220(),
        -1,
        {},
        {"rmr": -0.001118911788916191-0.0011739231139356607j},
        id="problem-20260220_r=-1",
    ),
]

@pytest.mark.parametrize(
    argnames="factory, r, func_kwargs, expected_results",
    argvalues=CASES,
)
def test_rightmost_root(factory, r, func_kwargs, expected_results, enable_plot: bool) -> None:
    """ Tests rightmost_root  """

    E, A, hA = factory

    rmr, meta = rightmost_root(E, A, hA, r=r, **func_kwargs)

    expected_rmr = expected_results.get("rmr", None)
    if expected_rmr:
        # note: we care only about real part as it is right-most root
        assert np.isclose(np.real(rmr), np.real(expected_rmr), rtol=1e-6, atol=1e-6)

    print(rmr, expected_rmr, rmr-expected_rmr)

    if enable_plot:
        raise NotImplementedError("Plotting not implemented yet for gamma_r tests.")
