"""
Tests for DDAE reprezentation

TODO:
    - split into multiple files later
"""

import pytest

import numpy as np
from scipy import linalg

import tdspy

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

    rdde = tdspy.DDAE(A=A, hA=hA)
    return rdde

def generate_example_02(a: float=0.25, tau1: float=1.0, tau2: float=2.0):
    """ Simple parametrized example from [1]
     
    E = dx(t)dt = A0 x(t) + A1(a) x(t-tau1) + A2 x(t-tau2)
    
    Args:
        a (float)
        tau1 (float)
        tau2 (float)
    
    Returns:
        ddea - DDAE representation of time-delay system

    Notes:
        Settings used in book [1]
            (a) a=0.25, tau1=1.0, tau2=2.0 - default
            (b) a=0.25, tau1=0.99, tau2=2.0
            (c) a=0.75, tau1=1.0, tau2=2.0
            (d) a=0.75, tau1=0.99, tau2=2.0

    References:
     [1] Michiels, W., & Niculescu, S., eds. Stability, control, and computation
         for time-delay systems: an eigenvalue-based approach. Society for
         Industrial and Applied Mathematics, 2014, page 35, equation (1.90)
    """
    E = np.array([[1, 0],
                  [0, 0.]])
    A0 = np.array([[0, -0.125],
                   [-1, 1]])
    A1 = np.array([[0, 0.],
                   [0, -a]])
    A2 = np.array([[0, 0.],
                   [0, 0.5]])
    
    A = np.stack([A0, A1, A2], axis=2)
    hA = np.array([0, tau1, tau2])

    rdde = tdspy.ddae.DDAE(E=E, A=A, hA=hA)
    return rdde

def test_compress():
    n = 4
    A = [np.eye(n) for i in range(5)]
    A.extend([np.eye(n), -np.eye(n)])
    A.extend([np.eye(n), -np.eye(n), np.eye(n)])
    hA = np.array([0 for _ in range(5)]+[1,1]+[2.5, 2.5, 2.5])
    A = np.stack(A, axis=2)
    ddae = tdspy.ddae.DDAE(E=np.eye(n), A=A, hA=hA)
    
    ddae.compress(inplace=True)

    assert np.all(ddae.A[:,:,0] == 5*np.eye(n))
    assert np.all(ddae.A[:,:,1] == np.eye(n))
    assert np.all(ddae.hA == np.array([0, 2.5]))

def test_create_02():
    ddae = generate_example_02()
    print(ddae.uE)
    print(ddae.vE)

def test_diff_02():
    ddae = generate_example_02()
    
    uE = ddae.uE
    vE = ddae.vE
    A = ddae.A

    D1 = np.zeros_like(A)
    norms = np.zeros(shape=(A.shape[2],))
    for i in range(A.shape[2]):
        Ai = A[:,:,i]
        Di = np.transpose(uE) @ Ai @ vE
        D1[:,:,i] = Di
        norms[i] = linalg.norm(Di, ord=1, axis=None)
    
    diff = ddae.get_delay_difference_equation()

    assert np.all(np.isclose(diff.A, D1, rtol=0, atol=1e-10))
    assert np.all(np.isclose(diff.E, 0, rtol=0, atol=1e-10))

    D2 = np.transpose(np.transpose(np.transpose(uE) @ A) @ vE)
    norm_mat = linalg.norm(D2, ord=1, axis=(0,1))
    print(f"{norm_mat=}")

    diff = ddae.get_delay_difference_equation()
    print(diff.A)
    print(diff.E)
    print(diff.uE)
    print(diff.vE)

def test_ddae_is_essentialy_retarded_or_neutral_01():
    ddae = generate_example_01()
    assert ddae.is_essentially_retarded
    assert not ddae.is_essentially_neutral

def test_ddae_is_essentialy_retarded_or_neutral_02():
    ddae = generate_example_02()
    assert not ddae.is_essentially_retarded
    assert ddae.is_essentially_neutral

def test_feedthrough_ddae():
    """ Creates a feedthrough DDAE """
    pass
