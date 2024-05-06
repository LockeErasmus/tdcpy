



def discretize(tds, N: int, s0: float=0, method: str="cheb"):
    """ Discretizes RDDE, NDDE or DDAE into DAE

    Discretizes Time-Delay System into Differential Algebraic Equation.

    Args:
        tds (RDDE, NDDE, DDAE): Definition of Time-Delay System
        N (int): degree of discretization N>0
        s0 (float): point discretization is done around, default 0
        method (str): type of approximation, default 'cheb', allowed 'cheb', 'legendre'

    Returns:
        TODO
    """
    
    assert isinstance(N, int) and N > 0, "N has to be int > 0"
    assert isinstance(s0, (int, float)), "s0 has to be float or int"
    assert method in ["cheb", "legendre"], "allowed methods from ['cheb', 'legendre']"
    
