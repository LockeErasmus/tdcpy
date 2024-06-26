"""
Example for gamma(r; normalized_delay_difference_eq)
"""
import logging

# Set up logging
import logging

import tdspy.stability.gamma_r
logger = logging.getLogger("tdspy")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

import numpy as np

import tdspy as tds
from tdspy.stability.gamma_r import compute_gamma, func

def generate_example() -> tds.NDDE:
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

def generate_example_2() -> tds.DDAE:
    A = np.stack([
        np.array([[0.25]]),
        np.array([[-1./3]]),
    ], axis=2)
    hA = np.array([0, 1.])
    H = np.stack([
        np.array([[-0.55]]),
        np.array([[1.25]]),
    ], axis=2)
    hH = np.array([0.1, 0.2])

    return tds.NDDE(A=A, hA=hA, H=H, hH=hH)


def generate_example_3() -> tds.NDDE:
    A = np.stack([
        np.array([[0.1, -0.2, -0.5],
                  [0, 2, -1.0],
                  [0, -1, -4]]),
        np.array([[0.1, 0.2, -0.5],
                  [2, 0, -0.8],
                  [0, -1, -1]]),
    ], axis=2)
    hA = np.array([0, 1.])
    H = np.stack([
        np.array([[0.1, 0.2, -0.5],
                  [0, 0, -1.0],
                  [0, 5, -4]]),
        np.array([[0.1, 0.2, -0.5],
                  [2, 0, -0.8],
                  [0, 5, -4]]),
        np.array([[0.1, 1.2, -0.5],
                  [2, 0, -1.0],
                  [-2, 5, -4]]),
    ], axis=2)
    hH = np.array([0.1, 0.2, 0.4])

    return tds.NDDE(A=A, hA=hA, H=H, hH=hH)

ndde = generate_example_3()
diff = ndde.get_delay_difference_equation()

g, info = compute_gamma(diff.A[:,:,1:], diff.hA[1:], 0, correction=True)







