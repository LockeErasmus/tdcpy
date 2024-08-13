"""
Delay controller design (stabilization via BFGS)
"""

import logging

import numpy as np
import numpy.typing as npt
from scipy import linalg, optimize

from tdspy.stability.characteristic_roots import rightmost_root, RightmostRootInfo

logger = logging.getLogger("__name__")

def func(x: npt.NDArray, E: npt.NDArray, P: npt.NDArray, hP: npt.NDArray, hK: npt.NDArray, Kmask: npt.NDArray, B: npt.NDArray, C: npt.NDArray):
        """
        
        The system is defined as

        E dxdt(t) = SUM P[i] x(t-hP[i]) + SUM B * K[j] * C x(t-hK[j])

        where K = x.reshape(Kshape)

        Args: TODO
            x
            E
            P
            hP
            hK
            Kshape
            Kmask
            B
            C

        Returns:
            tuple containing:

                - alpha (float): spectral abscissa
                - jac (array): jacobian        
        """

        # K = Vec(x) -> inverse operation
        K = x.reshape((B.shape[1], C.shape[0], hK.shape[0]))
        A = np.concatenate(
            [
                P,
                np.einsum(# more efficient way to obtain B @ K[:,:,i] @ C
                    'ijk,jn->ink',
                    np.einsum('ni,ijk->njk', B, K),
                    C,
                )
            ],
            axis=2,
        )
        hA = np.r_[hP, hK]
        rmr, rmr_info = rightmost_root(E, A, hA, r=0)

        conj_u_T = np.conj(rmr_info.u[np.newaxis,:]) # u* with shape=(1,n)
        v = rmr_info.v[:,np.newaxis] # v with shape=(n,1)
        dM = E + np.sum(A * hA * np.exp(-rmr*hA), axis=2) # dM(s)/ds evaluated at s=rmr
        den = conj_u_T @ dM @ v

        matrix = (conj_u_T @ B).T @ (C @ v).T
        gradient = np.real(1/den * (Kmask * hK) * matrix[:,:,np.newaxis])           # there's a bug here. 
        
        return np.real(rmr), gradient.reshape(-1)


def design_bfgs(E: npt.NDArray, P:npt.NDArray, hP:npt.NDArray, K0, hK, B, C, **kwargs):
    """
    Args:
        TODO
        **kwargs:
            mask (array): masking gradient
    """

    Kmask = kwargs.get("mask", np.full_like(K0, fill_value=True, dtype=bool))
    sol = optimize.minimize(
        func,
        K0.reshape(-1),
        args=(E, P, hP, hK, Kmask, B, C),
        jac=True,
        method="BFGS",
        options=kwargs.get("options", {}),
    )
    return sol


def gradient_test(func, x, h=1e-4, tolerance = 1e-6, **kwargs):
    """ function to test the numerical accuracy of the computed gradient using central differences
    The method uses central differences for testing the numerical gradient.
    
    Args:
        func:   cost function
        h:      step size
        E, P, hP, hK, Kmask, B, C:  arguments
    """      

    E = kwargs.get("E",np.eye(3))      
    P = kwargs.get("P",np.random.randint(10,size=(3,3,1)))
    hP = kwargs.get("hP",0.)
    hK = kwargs.get("hK",0.)
    Kmask = kwargs.get("Kmask",np.ones(shape=(1,1,1)))
    B = kwargs.get("B",np.array([[1,0,0]]).T)
    C = kwargs.get("C",np.array([[1,0,0]]))

    
    nvar = x.size

    _, g_analytical = func(x, E, P, hP, hK, Kmask, B, C)
    
    g_numerical = np.zeros_like(x)   # 

    for i in range(0,nvar):
        x_forward = np.copy(x)
        x_backward = np.copy(x)

        x_forward[i] += h
        x_backward[i] -= h

        f_forward, _ = func(x_forward,E, P, hP, hK, Kmask, B, C)
        f_backward, _ = func(x_backward,E, P, hP, hK, Kmask, B, C)

        g_numerical[i] = (f_forward - f_backward)/(2*h)
        
    diff = np.linalg.norm(g_analytical - g_numerical)

    if diff < tolerance:
         print("Gradient test passed!")

    else:
         print(f"Gradient test failed. Difference: {diff}")
         
    return g_analytical, g_numerical
