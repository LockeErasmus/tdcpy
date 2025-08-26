# Bugs/implementation issues

Refer the jupyter notebook coming_from_matlab.ipynb

2. Example 2.8: The plot displays incorrect results, compared with Figure 2.9
    Besides, if the discretization is increased, the function does not work correctly.

3. Example 2.12: Error in formulating the qp
    ValueError: could not broadcast input array from shape (2,) into shape (3,)
    issue in line 143 of quasipoly.py - looks incorrect to me!
    hH[-1, -1, :] = coefs[1:-1] 

4. test_common.py: The following tests fail
    1. test_static_controller_01(): 
        assert E.shape == (1,1)
        AssertionError: assert (1, 3) == (1, 1)
    2. test_dynamic_controller_01(): assert E.shape == ()
        assert E.shape == (2,2)
        AssertionError: assert (2, 4) == (2, 2)

5. concatenate_2x2_by_delays: What is the purpose of this function?
    I don't understand this function. 
    As per the documentation, it should be:
            E dxdt = A[:,:,0]*x(t-hA[0]) + ... + A[:,:,n] x(t-hA[n]) +
                 + B[:,:,0]*u(t-hB[0]) + ... + B[:,:,m] u(t-hB[n])
        
            y  = C[:,:,0]*x(t-hC[0]) + ... + C[:,:,p] x(t-hC[p]) +
                 + D[:,:,0]*u(t-hD[0]) + ... + D[:,:,q] u(t-hD[q])

            Concatenates the system into:

            E*dx1dt = A*[:,:,0]*x2(t-hA*[0]) + ... + A*[:,:,n*] x2(t-hA*[n*])

        where:
            x1 := [x^T y^T]^T
            x2 := [x^T u^T]^T
        and therefore:
            hA* = [hA, hB, hC, hD]
            n* = n+m+p+q
        left hand-side matrix:
            E* =    [E, 0]
                    [0, 0]
        right hand-side array:
            A*[:,:,:n] =        [A, 0]  
                                [0, 0]
            A*[:,:,n:n+m] =     [0, B]  
                                [0, 0]
            A*[:,:,n+m:n+m+p] = [0, 0]  
                                [C, 0]
            A*[:,:,n+m+p:] =    [0, 0]
                                [0, D]

        However, the function is incorrect to me. What happens to y?