import tdspy as tds
import numpy as np
import tdspy.plot as plt

# for closed-loop
from tdspy.common.composition import concatenate_2x2_by_delays
from tdspy.stability.characteristic_roots import rightmost_root, RightmostRootInfo
import tdspy.controller as controller
from tdspy.controller import create_dynamic_controller
from tdspy.common.compress import compress_matrices_delays



# Define system matrices
A0  = np.array([[0.2,0.1],[-0.5,1]])
A1  = np.array([[0.5,0.3],[0.1,-0.1]])
A   = np.stack([A0,A1],axis=2)
hA  = np.array([0.,1.])

B0  = np.eye(2)
B   = np.stack([B0],axis=2)
hB  = np.array([0.])

C0  = np.array([[1,1]])
C   = np.stack([C0],axis=2)
hC  = np.array([0.])

D   = np.stack([np.array([[0.01,0.01]])],axis=2)
hD  = np.array([0.])

tau = 1.

P = tds.DDAE(A=A,hA=hA,B=B,hB=hB,C=C,hC=hC,D=D,hD=hD)


# Define controller matrices
Ac = np.array([[-3.48]])
Bc = np.array([[3.1]])
Cc = np.array([[1.79],[-0.09]])
Dc = np.array([[-1.86],[-1.4]])

# Obtain matrix sizes
n   = np.shape(A0)[0]
p   = np.shape(B)[1]
q   = np.shape(C)[0]
nc  = np.shape(Ac)[0]

# Forming closed-loop
n_CL    = n+p+q+nc
E_CL    = np.zeros([n_CL,n_CL])
E_CL[:n,:n] = np.eye(n)
idx     = np.arange(n + q, n + q + nc)
E_CL[np.ix_(idx, idx)] = np.eye(nc)

A_CL_0  = np.zeros([n_CL,n_CL])
A_CL_0[:n,:n] = A0
A_CL_0[n:n+q,:n] = C[:,:,0]
idx     = np.arange(n,n+q)

A_CL_0[np.ix_(idx,idx)] = -np.eye(q)
A_CL_0[np.ix_(n+q+np.arange(0,nc),n+np.arange(0,q))] = Bc
A_CL_0[np.ix_(n+q+np.arange(0,nc),n+q+np.arange(0,nc))] = Ac
A_CL_0[np.ix_(n+q+nc+np.arange(0,p),n+np.arange(0,q))] = Dc
A_CL_0[np.ix_(n+q+nc+np.arange(0,p),n+q+np.arange(0,nc))] = Cc
A_CL_0[np.ix_(n+q+nc+np.arange(0,p),n+q+nc+np.arange(0,p))] = -np.eye(p)


A_CL_1 = np.zeros_like(A_CL_0)
A_CL_1[:n,:n] = A1
A_CL_1[np.ix_(np.arange(0,n),n+q+nc+np.arange(0,p))] = B[:,:,0]
A_CL_1[np.ix_(n+np.arange(0,q),n+q+nc+np.arange(0,p))] = D[:,:,0]

A_CL = np.stack([A_CL_0,A_CL_1],axis=2)
hA_CL = np.array([0.,tau])

CL = tds.DDAE(E=E_CL,A=A_CL,hA=hA_CL)

diff = CL.get_delay_difference_equation()

CD,_ = tds.spectral_abscissa_diff(diff)
print("CD=",CD)

# Compute rectangular spectral abscissa - works fine, commented out to save time
# l_rect,_ = tds.roots(CL,r=[-4,0.5,-500,500],max_size_evp=2000) 
sa = tds.spectral_abscissa(CL)
print("sp. abscissa is %.2f, must be -0.2845",sa)
# import matplotlib.pyplot as plt
# tds.plot.eigen_plot(l_rect)
# plt.show()

CL.print()
print(tds.strong_spectral_abscissa(CL))

# l_rhp,_ = tds.roots(CL,r=-3,max_size_evp=2000,discretization=15)
# print("l_rect=",np.max(np.real(l_rhp)))
# print("l_rhp=",l_rhp)
# plt.eigen_plot(l_rhp)


#--------------------OK until this point ------------------------------------------#
# -------------------Now using the closed-loop formulation ------------------------#

# Method 1: using controller.interconnect
# P: DDAE 
# K: DDAE

P = tds.DDAE(A=A,hA=hA,B=B,hB=np.array([1.]),C=C,hC=hC,D=D,hD=np.array([1.]))
K = tds.DDAE(A=np.stack([Ac],axis=2),hA=np.array([0.]),
             B=np.stack([Bc],axis=2),hB=np.array([0.]),
             C=np.stack([Cc],axis=2),hC=np.array([0.]),
             D=np.stack([Dc],axis=2),hD=np.stack([0.]))
hK = np.array([0.])

cl = controller.interconnect(P,K)           # this does not work!

# test measures
diff2 = cl.get_delay_difference_equation()
CD2,_ = tds.spectral_abscissa_diff(diff2)

diff2.print()

print("Closed-loop 1 is neutral: ",CL.is_essentially_neutral)
print("Closed-loop 2 is neutral:", cl.is_essentially_neutral)



# Method 2: using tds.closed_loop
# P: DDAE
# cont: DDAE

cont = tds.controller.create_dynamic_controller(Ac,Bc,Cc,Dc)

E, K1, hK1 = concatenate_2x2_by_delays(cont.E, cont.A, cont.B, cont.C, cont.D, cont.hA, cont.hB, cont.hC, cont.hD)
K1, hK1 = compress_matrices_delays(K1, hK1)

cl2 = tds.ClosedLoop(P, order=1,y_indices=[0],u_indices=[0,1],K0=K1, hK=hK1)
print(cl2.A[:,:,0])         # this is not correct!

#_--------------------------NOTES----------------------------#
# The D11 matrix block and the B11 matrix block from the original system are seen in the tau=0 blocks
# It should appear in the tau1 block
# Reason for the problem:
# in line 128 of tds.ClosedLoop, it calls the function concatenate_2x2_by_delays
# In the arguments, the B, C and D matrices are all assigned to the zero delay matrix of A
# if you check self.system.hB, self.system.hD it shows 0

cl2.print()
# cl2.is_essentially_neutral
