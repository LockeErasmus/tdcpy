"""
Tests for common module

TODO:
    - split into multiple files later
"""

import pytest

import numpy as np
from scipy import linalg

from tdspy.common.compress import compress_matrices_delays


def test_compression_empty():
    """ Tests compression of empty representation """

    A = np.zeros(shape=(4,4,0))
    hA = np.zeros(shape=(0,))

    compressed_A, compressed_hA = compress_matrices_delays(A, hA)

    assert A.shape == compressed_A.shape
    assert hA.shape == compressed_hA.shape
    assert compressed_A.size == 0
    assert compressed_hA.size == 0

def test_compression_01():
    """ Tests compression """

    n = 4
    A = [np.eye(n) for i in range(5)]
    A.extend([np.eye(n), -np.eye(n)])
    A.extend([np.eye(n), -np.eye(n), np.eye(n)])
    hA = np.array([0 for _ in range(5)]+[1,1]+[2.5, 2.5, 2.5])
    A = np.stack(A, axis=2)

    compressed_A, compressed_hA = compress_matrices_delays(A, hA)

    assert np.all(compressed_A[:,:,0] == 5*np.eye(n))
    assert np.all(compressed_A[:,:,1] == np.eye(n))
    assert np.all(compressed_hA == np.array([0, 2.5]))







