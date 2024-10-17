"""
Tests quasipolynomial representation
"""

import pytest

import numpy as np
from scipy import linalg

import tdspy

from tdspy.common.quasipoly import compress_qp, qp_to_ndde

def generate_example_00():
    delays = np.array([0.0, 1.0])
    coefs = np.array([[1., 0],[0, 0]])
    return coefs, delays, True

def generate_example_01():
    delays = np.array([0.0, 1.0])
    coefs = np.array([[0, 1],[1, 0]])
    return coefs, delays, True

def generate_example_02():
    """ Example from tds-control MATLAB package """
    delays = np.array([0., 1, 2])
    coefs = np.array([
        [1, -2, 1.],
        [0, -2, 2],
        [0, 0, 1],
    ])
    return coefs, delays, False

def test_qp_to_ndde_00():
    coefs, delays, ascending = generate_example_00()
    A, hA, H, hH = qp_to_ndde(coefs, delays, ascending=ascending)
    
    with np.printoptions(precision=4, linewidth=1000, suppress=True):
        for i in range(hA.shape[0]):
            print(f"A[:,:,{i} - tau={hA[i]}")
            print(A[:,:,i])
            print("-"*50)

        for i in range(hH.shape[0]):
            print(f"H[:,:,{i} - tau={hA[i]}")
            print(H[:,:,i])
            print("-"*50)

def test_qp_to_ndde_01():
    coefs, delays, ascending = generate_example_01()
    A, hA, H, hH = qp_to_ndde(coefs, delays, ascending=ascending)
    
    with np.printoptions(precision=4, linewidth=1000, suppress=True):
        for i in range(hA.shape[0]):
            print(f"A[:,:,{i} - tau={hA[i]}")
            print(A[:,:,i])
            print("-"*50)

        for i in range(hH.shape[0]):
            print(f"H[:,:,{i} - tau={hA[i]}")
            print(H[:,:,i])
            print("-"*50)

def test_qp_to_ndde_02():
    coefs, delays, ascending = generate_example_02()
    
    # correct arrays
    correct_hA = np.array([0., 1, 2])
    A0 = np.array([
        [ 0., 1.],
        [-1., 2.],
    ])
    A1 = np.array([
        [ 0.,  0.],
        [-2.,  2.],
    ])
    A2 = np.array([
        [ 0., 0.],
        [-1., 0.],
    ])
    correct_A = np.stack([A0, A1, A2], axis=2)

    correct_hH = np.array([])
    correct_H = np.zeros(shape=(2,2,0))
    
    A, hA, H, hH = qp_to_ndde(coefs, delays, ascending=ascending)

    assert np.allclose(A, correct_A)
    assert np.allclose(hA, correct_hA)
    assert H.size == 0 and H.shape == correct_H.shape
    assert hH.size == 0 and hH.shape == correct_hH.shape

    with np.printoptions(precision=4, linewidth=1000, suppress=True):
        for i in range(hA.shape[0]):
            print(f"A[:,:,{i} - tau={hA[i]}")
            print(A[:,:,i])
            print("-"*50)

        for i in range(hH.shape[0]):
            print(f"H[:,:,{i} - tau={hA[i]}")
            print(H[:,:,i])
            print("-"*50)