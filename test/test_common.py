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

def test_sort_empty():
    """ Tests sort of empty representation """

    A = np.zeros(shape=(4,4,0))
    hA = np.zeros(shape=(0,))

    sorted_A, sorted_hA = compress_matrices_delays(A, hA)

    assert A.shape == sorted_A.shape
    assert hA.shape == sorted_hA.shape
    assert sorted_A.size == 0
    assert sorted_hA.size == 0

def test_sort_01():
    """ Tests sort of empty representation """

    n = 4
    A = [i*np.eye(n, dtype=np.float64) for i in range(1,6)]
    A = np.stack(A, axis=2)
    hA = np.array([0.1, 1.5, 0.05, 0.2, 0.8])

    sorted_A, sorted_hA = compress_matrices_delays(A, hA)

    assert A.shape == sorted_A.shape
    assert hA.shape == sorted_hA.shape
    assert np.allclose(sorted_hA, np.array([0.05, 0.1, 0.2, 0.8, 1.5]))
    assert np.allclose(sorted_A, A[:,:,[2, 0, 3, 4, 1]])
