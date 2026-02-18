# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Adam Peichl
# Copyright (C) 2026 Adrian Saldanha

"""
Tests for NDDE representation

TODO:
    - split into multiple files later
"""

import numpy as np

import tdcpy as tds

def generate_example_01() -> tds.NDDE:
    """ generates example from TDS MATLAB manual (page 23) """

    A = np.stack([
        np.array([[0.25]]),
        np.array([[-1./3]]),
    ], axis=2)
    hA = np.array([0, 1.])
    H = np.stack([
        np.array([[-0.75]]),
        np.array([[0.25]]),
    ], axis=2)
    hH = np.array([1., 2])

    return tds.NDDE(A=A, hA=hA, H=H, hH=hH)

def test_conversion_to_ddae():
    ndde = generate_example_01()

    ddae = ndde.to_ddae()

    assert np.allclose(ddae.E, np.array([[0,1],[0,0.]]), atol=1e-10)
    assert ddae.A.shape == (2,2,4)
    assert np.allclose(ddae.A[:,:,0], np.array([[0.25,0],[1,-1]]), atol=1e-10)
    assert np.allclose(ddae.A[:,:,1], np.array([[-1./3,0],[0,0]]), atol=1e-10)
    assert np.allclose(ddae.A[:,:,2], np.array([[0,0],[-0.75,0]]), atol=1e-10)
    assert np.allclose(ddae.A[:,:,3], np.array([[0,0],[0.25,0]]), atol=1e-10)
    assert ddae.hA.shape == (4,)
    assert np.allclose(ddae.hA, np.array([0,1,1,2.]), atol=1e-10)

    ddae.compress(inplace=True)

    assert np.allclose(ddae.E, np.array([[0,1],[0,0.]]), atol=1e-10)
    assert ddae.A.shape == (2,2,3)
    assert np.allclose(ddae.A[:,:,0], np.array([[0.25,0],[1,-1]]), atol=1e-10)
    assert np.allclose(ddae.A[:,:,1], np.array([[-1./3,0],[-0.75,0]]), atol=1e-10)
    assert np.allclose(ddae.A[:,:,2], np.array([[0,0],[0.25,0]]), atol=1e-10)
    assert ddae.hA.shape == (3,)
    assert np.allclose(ddae.hA, np.array([0,1,2.]), atol=1e-10)

    assert ddae.is_compressed



def test_ddae_ndde_conversion_equal():
    """
    
    Tests if paths
        (1) ndde -> diff
        (2) ndde -> ddae -> diff -> normalized diff
    Results in the same solution
    
    """
    ndde = generate_example_01()

    ddae = ndde.to_ddae() # this step should be correct
    ddae.compress(inplace=True)

    print("uE.T  ", ddae.uE.T)
    print("vE    ", ddae.vE)

    for i in range(ddae.A.shape[2]):
        print(f"{i=}  ", ddae.uE.T @ ddae.A[:,:,i] @ ddae.vE)

    diff1 = ddae.get_delay_difference_equation()
    print(diff1.A.shape)
    print(diff1.A)
    
    diff2 = ndde.get_delay_difference_equation()
    # print(diff2.A[:,:,0])
    # print(diff2.A[:,:,0])




    

    