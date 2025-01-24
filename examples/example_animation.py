
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import tdspy as tds
import tdspy.plot as tdsplot


# Set up logging
import logging
logger = logging.getLogger("tdspy")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

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
hA = np.array([0,1.])
r = -2.5

rdde = tds.DDAE(A=A, hA=hA)


ani = tdsplot.discretization_animation(rdde, s0=0, discretization=range(10,85), discretization_ideal=100, xlim=(-3.25,1.5), ylim=(-200, 200))

writer = animation.PillowWriter(fps=30,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
ani.save('scatter.gif', writer=writer)

plt.show()