

import tdcpy
import numpy as np
import matplotlib.pyplot as plt


def create_ddae(epsilon: float) -> tdcpy.DDAE:

    sinc_epsilon = np.sin(epsilon)/epsilon

    E = np.array([
        [1],
    ])

    A0 = np.array([[0.]])

    B0 = np.array([
        [0],
    ])
    B1 = np.array([
        [1],
    ])
    B2 = np.array([
        [-1],
    ])
    hB = np.array([0, np.pi-epsilon, np.pi+epsilon])

    C0 = 1 / (1+sinc_epsilon) * np.array([[1./2/epsilon]])

    D0 = 1 / (1+sinc_epsilon) * np.array([[sinc_epsilon]])

    print(C0, D0)

    # print(E.shape, A0.shape, B0.shape, B1.shape, B2.shape, hB.shape, C0.shape, D0.shape)

    ddae = tdcpy.DDAE(
        [A0], np.array([0.]), E,
        [B0, B1, B2], hB,
        [C0], np.array([0.]),
        [D0], np.array([0.])
    )
    return ddae


import tdcpy.plot
for epsilon in np.linspace(0.1, 0.2, 5):
    region = [-10, 10, -50, 50]
    ddae = create_ddae(epsilon)
    r, meta = tdcpy.zeros(ddae, region)
    print(meta.discretization)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    # tdcpy.plot.eigen_plot(r, ax=ax)
    # tdcpy.plot.complex_scatter_axplot(meta.newton_unconverged_initial_guesses, ax=ax, color="red", marker="o", alpha=0.1)
    tdcpy.plot.eigen_plot(r, ax=ax1)
    ax1.set_title(f"Transmission zeros for epsilon={epsilon:.2f}")

    ax2.set_title(f"Transmission zeros for epsilon={epsilon:.2f} (detail)")
    tdcpy.plot.eigen_plot(r, ax=ax2)
    ax2.set_xlim((-1, 1))
    ax2.set_ylim((-10, 10))

    tdcpy.plot.eigen_plot(meta.discretization_eigenvalues, ax=ax3)
    ax3.set_title(f"Discretization eigenvalues for N={meta.discretization}")

    tdcpy.plot.eigen_plot(meta.discretization_eigenvalues, ax=ax4)
    ax4.set_title(f"Discretization eigenvalues for N={meta.discretization} (detail)")
    ax4.set_xlim(region[0:2])
    ax4.set_ylim(region[2:4])



    plt.show()