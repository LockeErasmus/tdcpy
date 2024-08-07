"""
Delay controller design (stabilization via BFGS)
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg, optimize

from tdspy.stability.characteristic_roots import rightmost_root, RightmostRootInfo


logger = logging.getLogger("__name__")

def func(x: npt.NDArray, E, P, hP, hK, Kshape, Kmask, B, C):
        
        K = x.reshape(Kshape)
        A = np.concatenate(
            [
                P,
                np.stack([B @ K[:,:,i] @ C for i in range(K.shape[2])], axis=2)
            ],
            axis=2,
        )
        hA = np.r_[hP, hK]
        rmr, rmr_info = rightmost_root(E, A, hA, r=0)

        conj_u_T = np.conj(rmr_info.u[np.newaxis,:])
        v = rmr_info.v[:,np.newaxis]
        dM = E + np.sum(A * hA * np.exp(-rmr*hA), axis=2)
        coef = conj_u_T @ dM @ v

        matrix = (conj_u_T @ B).T @ (C @ v).T
        gradient = np.real(1/coef * (Kmask * hK) * matrix[:,:,np.newaxis])
        
        return np.real(rmr), gradient.reshape(Kshape)


def design_bfgs(E: npt.NDArray, P:npt.NDArray, hP:npt.NDArray, K0, hK, B, C, **kwargs):
    """
    Args:
        TODO
        **kwargs:
            mask (array): masking gradient
    """

    Kmask = kwargs.get("mask", np.full_like(K0, fill_value=True, dtype=bool))
    Kshape = K0.shape

    def func(x: npt.NDArray):
        
        K = x.reshape(Kshape)
        A = np.concatenate(
            [
                P,
                np.stack([B @ K[:,:,i] @ C for i in range(K.shape[2])], axis=2)
            ],
            axis=2,
        )
        hA = np.r_[hP, hK]
        rmr, rmr_info = rightmost_root(E, A, hA, r=0)

        conj_u_T = np.conj(rmr_info.u[np.newaxis,:])
        v = rmr_info.v[:,np.newaxis]
        dM = E + np.sum(A * hA * np.exp(-rmr*hA), axis=2)
        coef = conj_u_T @ dM @ v

        matrix = (conj_u_T @ B).T @ (C @ v).T
        gradient = np.real(1/coef * (Kmask * hK) * matrix[:,:,np.newaxis])
        
        return np.real(rmr), gradient.reshape(-1)
    
    sol = optimize.minimize(
        func,
        K0.reshape(-1),
        jac=True,
        method="bfgs",
        options=kwargs.get("options", {}),
    )
    return sol







