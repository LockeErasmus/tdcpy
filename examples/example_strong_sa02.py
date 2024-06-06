"""

"""
import numpy as np
import matplotlib.pyplot as plt
import tdspy as tds

np.random.seed(100)

A = np.random.rand(5,5,5)
hA = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
E = np.zeros(shape=(5,5))
E[[0,1], [0,1]] = 1

if __name__ == "__main__":
    # Set up logging
    import logging
    logger = logging.getLogger("tdspy")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    ddae = tds.DDAE(A=A, hA=hA, E=E)

    print(f"The DDAE is essentially neutral={ddae.is_essentially_neutral}")

    cd = tds.cd(ddae)

    print(f"strong spectral abscissa of associated DIFF {cd=}")

