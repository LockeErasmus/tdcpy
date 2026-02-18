"""

"""
import numpy as np
import matplotlib.pyplot as plt
import tdcpy as tds

np.random.seed(100)

A = np.random.rand(5,5,5)
hA = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
E = np.zeros(shape=(5,5))
E[[0,1], [0,1]] = 1

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

if __name__ == "__main__":
    # Set up logging
    import logging
    logger = logging.getLogger("tdcpy")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    #ddae = tds.DDAE(A=A, hA=hA, E=E)

    ndde = generate_example()

    #print(f"The DDAE is essentially neutral={ddae.is_essentially_neutral}")

    cd, info = tds.cd(ndde)

    if False:
        for method in ["hybr", "lm", "broyden1", "broyden2", "anderson", "linearmixing",
                    "diagbroyden", "excitingmixing", "krylov", "df-sane"]:
            try:
                cd = tds.cd(ddae, scipy_root_method=method)
            except:
                print(f"{method=} FAILED")

    print(f"strong spectral abscissa of associated DIFF {cd=}")

