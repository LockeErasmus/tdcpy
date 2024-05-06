"""
Tests for DDAE reprezentation

TODO:
    - split into multiple files later
"""

import numpy as np

import tdspy.ddae

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

    rdde = tdspy.ddae.DDAE(A=[A0, A1], hA=[0, 1.])
    return rdde

def generate_example_02():
    # Create DDAE representation
    A0 = np.array([[-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -10, -4],
                [0, 0, 4, -10]])
    A1 = np.array([[3, 3, 3, 3],
                [0, -1.5, 0, 0],
                [0, 0, 3, -5],
                [0, 5, 5, 5]])

    rdde = tdspy.ddae.DDAE(A=[A0, A1], hA=[0, 1.])
    return rdde

def test_dummy():
    pass


def test_foo():
    rdde = generate_example_01()
    print(rdde.uE)
    print(rdde.vE)

def test_diff_01():
    rdde = generate_example_01()
    diff = rdde.to_delay_difference_equation()
    print(diff.A)
    print(diff.E)
    print(diff.uE)
    print(diff.vE)