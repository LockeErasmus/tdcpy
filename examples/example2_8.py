import tdspy as tds
import numpy as np
import tdspy.plot as plt

tds.init_logger(level="WARNING")


#--------------------Beginning of Example 2.6----------------------#

# Define parameters
tau1, tau2 = 1., 2.

# Define matrices A0, A1
A0 = np.array([[1/4]])
A1 = np.array([[-1/3]])
A = np.stack([A0,A1],axis=2)
hA = np.array([0,tau1])

# Define matrices H1, H2
H1 = np.array([[-3/4]])
H2 = np.array([[1/2]])
H = np.stack([H1,H2],axis=2)
hH = np.array([tau1,tau2])

# Build TDS
ndde = tds.NDDE(H=H,hH=hH,A=A,hA=hA)
# ddae = ndde.to_ddae()

r=[-0.9,0.2,-500,500]

cr, RootsInfo = tds.roots(ndde,r,max_size_evp=1500)

tds.plot.eigen_plot(cr) # fig 2.5

# Now change the second delay to tau2=2.05

tau2 = 2.05
ndde.hH[1] = tau2
cr2, RootInfo = tds.roots(ndde,r,max_size_evp=1500) 

tds.plot.eigen_plot(cr2)    # fig 2.6
ndde.hH[1] = 2.05

# now change it to 2.005

tau2 = 2.005
ndde.hH[1] = 2.005

ndde.hH[1] = tau2
r = [-0.9,0.2,-500,500]
cr3, RootInfo = tds.roots(ndde,r,max_size_evp=1500) 

tds.plot.eigen_plot(cr3)  # fig 2.7

# Let's check the underlying delay difference equation

delay_diff = ndde.get_delay_difference_equation()

cr4, RootsInfo = tds.roots(delay_diff,r,max_size_evp=1500)  
tds.plot.eigen_plot(cr4)  # fig 2.8


#--------------------Beginning of Example 2.7----------------------#

c = tds.strong_spectral_abscissa(ndde,-0.2)

gamma0 = tds.gamma(ndde,0)


#-------------------Beginning of Example 2.8----------------------#
r = [-0.9,0.2,-200,200]
tau2 = 2
ndde.hH[1] = tau2
cr5, RootInfo = tds.roots(ndde,r)
import matplotlib.pyplot as plt
tds.plot.eigen_plot(cr5)     # fig 2.9
plt.show()  
np.max(np.real(cr5))


# # the below code does not work if discretization is > 10 #
tau2 = 2
ndde.hH[1] = tau2
cr6, RootInfo = tds.roots(ndde,-0.6, discretization=20)
import matplotlib.pyplot as plt
tds.plot.eigen_plot(cr6)
plt.show()  

