r"""

"""

import numpy as np
import tdcpy

from tdcpy.stability.characteristic_roots import rightmost_root

tdcpy.init_logger("DEBUG")

Th, Ta,Td, Tc = 14, 3, 3, 25
Kb, Ka, Kd, Kc, Ku = 0.24, 1, 0.94, 0.81, 0.39
nh, tb, te, td = 6.5, 40, 13, 18
tc, nc, tu = 2.8, 9.2, 13.2

hA, hB = np.array([0, nh, tb, te, td, tc, nc]), np.array([tu])

# A0
A0 = np.zeros(shape=(5, 5))
A0[1,0] = Ka / Ta
A0[1,1] = (-Ka-1) / Ta
A0[2,2] = -1 / Td
A0[4,3] = -1 

# A1
A1 = np.zeros(shape=(5, 5))
A1[0,0] = -1/Th

# A2
A2 = np.zeros(shape=(5, 5))
A2[0,1] = Kb/Th

# A3
A3 = np.zeros(shape=(5, 5))
A3[1,3] = 1/Ta

# A4
A4 = np.zeros(shape=(5, 5))
A4[2,1] = Kd/Td

# A5
A5 = np.zeros(shape=(5, 5))
A5[3,2] = Kc/Tc

# A6
A6 = np.zeros(shape=(5, 5))
A6[3,3] = -1/Tc

# Input matrix B
B0 = np.array([[Ku/Th], [0.], [0.], [0.], [0.]])

# Output matrix C
C0 = np.eye(5)
hC = np.array([0.])

# Output matrix D
D = np.zeros(shape=(5,1), dtype=float)
hD = np.array([0.])


plant = tdcpy.DDAE(
    A=[A0, A1, A2, A3, A4, A5, A6] , hA=hA,
    B=[B0], hB=hB,
    C=[C0], hC=hC,
    D=[D], hD=hD,
)

tdcpy.utils.print_system_matrices(plant)

Dc1 = np.array([0.35936, 1.2544, 3.1696, 3.9919, 0.14344])

Dc1 = np.array([-0.4838 , -2.3657, -3.8564, -4.9999, 0.2037])

# Dc1 = np.array([-0.1659 􀀀0:2968 􀀀0:3612 􀀀0:3629 0:0168

A7 = np.zeros(shape=(5, 5))
A7[0,:] = Ku/Th * Dc1

# plant = tdcpy.RDDE(
#     A=[A0, A1, A2, A3, A4, A5, A6, A7] , hA=np.r_[hA, hB],
# )

# cr, info = tdcpy.roots(plant, r=-0.15)

# BB = np.array([[1.], [0], [0], [0.]])
# CC = np.eye(5)

# print(CC)
# # print(BB @ (Ku/Th * Dc1[None, :]) )
# print(BB @ (Ku/Th * Dc1[None, :]) @ CC)

# print(A7)


Dc = np.zeros(shape=(1,5))

E = np.eye(*A0.shape)
A = np.stack([A0, A1, A2, A3, A4, A5, A6, A7], axis=2)
hA = np.r_[hA, hB]

for i in range(0):
    A[0:1, :, -1] = Dc
    
    rmr, rmr_info = rightmost_root(E, A, hA, r=0)
    
    # calculate gradient
    conj_u_T = np.conj(rmr_info.u[np.newaxis,:]) # u* with shape=(1,n)
    v = rmr_info.v[:,np.newaxis] # v with shape=(n,1)
    dM = rmr_info.DM # dM(s)/ds evaluated at s=rmr
    den = conj_u_T @ dM @ v # gradient denumenator
    matrix = (conj_u_T @ B0).T @ v.T

    # only real part, see spectral abscissa definition
    sa = np.real(rmr)
    sa_grad = np.real(1/den * (np.exp(-rmr*tu)) * matrix)

    # fgrad = np.real(1/den * (np.exp(-rmr*hK)) * matrix[:,:,np.newaxis])


    # print(rmr, rmr_info)

    # print(sa, sa_grad)

    Dc[:,:] -= 0.1 * sa_grad

    print(f"{i}| {sa=}   | Dc={Dc[0]}")



from scipy import optimize

def func(x):
    A[0, :, -1] = x
    rmr, rmr_info = rightmost_root(E, A, hA, r=0)
    
    # calculate gradient
    conj_u_T = np.conj(rmr_info.u[np.newaxis,:]) # u* with shape=(1,n)
    v = rmr_info.v[:,np.newaxis] # v with shape=(n,1)
    dM = rmr_info.DM # dM(s)/ds evaluated at s=rmr
    den = conj_u_T @ dM @ v # gradient denumenator
    matrix = (conj_u_T @ B0).T @ v.T

    # only real part, see spectral abscissa definition
    sa = np.real(rmr)
    sa_grad = np.ravel(np.real(1/den * (np.exp(-rmr*tu)) * matrix))

    return sa, sa_grad

sol = optimize.minimize(
    func,
    np.zeros(5),
    jac=True,
    method="BFGS",
    # options=kwargs.get("options", {}),
    # callback=kwargs.get("callback", None)
)

print(sol.x)




# import matplotlib.pyplot as plt
# import tdcpy.plot

# tdcpy.plot.eigen_plot(cr)
# plt.show()

