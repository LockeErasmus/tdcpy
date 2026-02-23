import numpy as np
import tdcpy as tds
from tdcpy.stability.spectral_abscissa import spectral_abscissa_diff
from tdcpy.common.delay_difference_equation import normalize_diff
from tdcpy.stability.gamma_r import gamma_normalized_diff

import logging
logger = logging.getLogger("tdcpy")
logger.setLevel("DEBUG")

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

# Create a formatter and attach it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)


#################### CASE I ###############

E = np.zeros(shape=(4,4))
A = np.zeros(shape=(4,4,4))

A[:,:,0] = np.array([[     0,    -1,     0,     0],
                     [    -1,     0,     0,     0],
                    [     0,     0,     0,    -1],
                    [     0,     0,    -1,     0]])

A[:,:,1] = np.array([   [      0,     0,     0,     0],
                        [      0,     3,     0,     0],
                        [      0,     4,     0,     0],
                        [      0,     1,     0,     0]])

A[:,:,2] = np.array([[     0,     0,     0,     0],
                    [      0,   0.4000,     0,     0],
                    [      0,  -0.4000,     0,     0],
                    [      0,  -0.4000,     0,     0]])

A[:,:,3] = np.array([[     0.01,     0,     0.01,     0.01],
                    [      0,   0,     0,     0],
                    [      0,  0,     0,     0],
                    [      0,  0,     0,     0]])

hA = np.array([0., 2.5, 5., 0.])

diff = tds.DDAE(E=E,A=A,hA=hA)

DD, hDD = normalize_diff(diff.A, diff.hA)

g0, g0_info = gamma_normalized_diff(DD, hDD, r=0)

print(f"gamma_0: {g0}") # OK

val,info = tds.spectral_abscissa_diff(diff, r=-1)
print(f"SA of diff: {val}") # must be -0.8657


########### CASE II ############

# Define E as a NumPy array
E = np.array([
    [1., 0., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0.]
])

# Define A as a 3D NumPy array of shape (7, 7, 4)
A = np.zeros((7, 7, 4))

# Fill in the slices of A
A[:, :, 0] = np.array([
    [-0.08, -0.03,  0.2,   0.,   0.,   0.,   0.],
    [ 0.2,  -0.04, -0.005, 0.,   0.,   0.,   0.],
    [-0.06,  0.2,  -0.07,  0.,   0.,   0.,   0.],
    [ 1.,    0.,    0.,    0.,   0.,   0.,  -1.],
    [ 0.,    1.,    0.,    0.,   0.,  -1.,   0.],
    [ 0.,    0.,    1.,    0.,  -1.,   0.,   0.],
    [ 0.,    0.,    0.,   -1.,   0.,   0.,   0.]
])

A[:, :, 1] = np.array([
    [0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 3., 0., 0., 0.],
    [0., 0., 0., 4., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0.]
])

A[:, :, 2] = np.array([
    [ 0.,   0.,   0.,  -0.1,  0.,   0.,   0.],
    [ 0.,   0.,   0.,  -0.2,  0.,   0.,   0.],
    [ 0.,   0.,   0.,   0.1,  0.,   0.,   0.],
    [ 0.,   0.,   0.,   0.4,  0.,   0.,   0.],
    [ 0.,   0.,   0.,  -0.4,  0.,   0.,   0.],
    [ 0.,   0.,   0.,  -0.4,  0.,   0.,   0.],
    [ 0.,   0.,   0.,   0.,   0.,   0.,   0.]
])

A[:, :, 3] = np.array([
    [0., 0., 0., 0.,    0.,    0.,    0.],
    [0., 0., 0., 0.,    0.,    0.,    0.],
    [0., 0., 0., 0.,    0.,    0.,    0.],
    [0., 0., 0., 0.,    0.,    0.,    0.],
    [0., 0., 0., 0.,    0.,    0.,    0.],
    [0., 0., 0., 0.,    0.01,  0.01,  0.01], # different from CASE 1
    [0., 0., 0., 0.,    0.,    0.,    0.],
])

# Define hA as a NumPy array
hA = np.array([0., 2.5, 5., 0.])

ddae = tds.DDAE(E=E,A=A,hA=hA)
diff = ddae.get_delay_difference_equation()

DD, hDD = normalize_diff(diff.A, diff.hA)

g0, g0_info = gamma_normalized_diff(DD, hDD, r=0)
print(f"gamma_0: {g0}")

val,info = tds.spectral_abscissa_diff(diff, r=-1) # overflow in eigenvalue computation
print(f"SA of diff: {val}") # must be -inf