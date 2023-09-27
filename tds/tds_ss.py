
import numpy as np
import numpy.typing as npt


class TDS:
    """ State-space representation for LTI time-delay systems """


    def __init__(self, E: npt.NDArray, A=list[npt.NDArray], hA=list[float]) -> None:
        
        assert len(A) > 0, f"At least one A matrix is expected"
        assert len(A) == len(hA), "Same number of A and delays"
        # TODO assert all elements hA are unique

        # sort delays and matrices of dynamics (delays ascending order)
        sorted_pairs = sorted(zip(A, hA))
        self._A, self._hA = [list(t) for t in zip(*sorted_pairs)]
        self._E = E

        # TODO compress delays - no duplicates

        # TODO add A0=zeros() and hA = [0, hA] if there is no 0.0 delay
        
        
def tds_roots(tds: TDS, region: float, **options):
    1

    # unpack values
    n = len(tds._hA)# number of delays
    E = tds._E
    A = tds._A
    hA = tds._hA

    # CASE 1 : TODO - no delay --> ODE

    # CASE 2 : TODO -> DDE
    mA = len(hA) # number of delays

    ## re-scale that max(hA) == 1.0
    tau_s = [d/hA[-1] for d in hA]
    K = []
    for a in A:
        K.append(hA[-1]*a)
    


    # heuristics for N (degree of spectral discretization)
    max_size_evp =  600 # TODO into **options
    N_max = int(max_size_evp / n) - 1 # condition: (N+1)*n <= max_size_evp

    if type(region) in [float, int]: # TODO RHP case: compute roots in a specified RHP
        rs = region * hA[-1] # rescale r
        # introduce a shift of the origin, shifted matrices are stored in B and C
        B = [k + -rs*E for k in K]
        C = np.zeros(shape=(n, n, mA-1))
        for i in range(mA-1):
            C[:,:,i] = K[i+1]*np.exp((-rs) * tau_s[i+1])
        










