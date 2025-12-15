"""
Tests for RDDE representation

TODO:
    - split into multiple files later
"""

import numpy as np

import tdspy as tds

def generate_example_01():
    # Create DDAE representation
    A0 = np.array([[-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -10, -4],
                [0, 0, 4, -10]])
    A1 = np.array([[3, 3, 3, 3],
                [0, -1.5, 0, 0],
                [0, 0, 3, -5],
                [0, 5, 5, 5]])
    A = np.stack([A0, A1], axis=2)
    hA = np.array([0, 1.])

    rdde = tds.RDDE(A=A, hA=hA)
    return rdde

def test_init_01():

    rdde = generate_example_01()

    # tests
    assert rdde.n == 4
    assert np.all(rdde.E == np.eye(4))


def test_to_ddae_01():
    rdde = generate_example_01()
    assert isinstance(rdde, tds.RDDE)
    ddae = tds.DDAE(A=rdde.A, hA=rdde.hA)
    print(ddae.is_essentially_retarded)



