Project structure and hierarchy
-------------------------------
-------------------------------

TDSpy package
-------------------------------



* `docs`
    * `coming_from_matlab.md`
* `examples`
* `src`
    * [`tdspy`](#tdspy)
        * [`common`](#common)
            * [`closed_loop`](#closed-loop)
            * [`composition`](#composition)
            * [`compress`](#compress)
            * [`delay_difference_equation`](#compress)
            * [`discretization`](#discretization)
        * [`plot`](#plot)
            * `init`
            * `eigenvalues`
            * `discretization_animation`
        * [`stability`](#stability)
            * `bounds`
            * `characteristic_roots`
            * `discretization_heuristic`
            * `gamma_r`
            * `newton`
            * `spectral_abscissa`
        * [`stabopt`](#stabopt)
            * `controller_bfgs`
        * [`base`](#base)
        * [`controller`](#controller)
        * [`dae`](#dae)          
        * [`ddae`](#ddae)
        * [`gamma`](#gamma)    
        * [`ndde`](#ndde)
        * [`rdde`](#rdde)     
        * [`roots`](#roots)    
        * [`zeros`](#zeros)    
        * [`spectral_abscissa`](#spectral_abscissa)
* `test`
* `.gitignore`
* `LICENSE`
* `pyproject.toml`
* `README.md`

### Modules

#### `tdspy`

##### `common`

###### `delay_difference_equation`:
    Set of functions for obtaining and manipulating the delay-difference equations
- ddae_to_diff: Converts delay    differential algebraic equation (DDAE) to delay-difference equation
    
    Syntax: 

        E*dx/dt = A[0] x(t-hA[0]) + ... + A[m-1] x(t-hA[m-1])
        
        ddae_to_diff(E, A, hA)

- ndde_to_diff(H, hH): Converts NDDE to delay difference equation

    For a NDDAE, the associated delay difference equation is given by
        
    Syntax:

            I*x(t) + H[0]*x(t-hH[0]) + ... + H[mH]*x(t-hH[mH]) = 0  

            ndde_to_diff(H, hH)

- normalize_diff(D: npt.NDArray, hD: npt.NDArray): 

    Normalizes delay difference equation

    Transforms the delay difference equation such that the leading zero delay
    matrix D[0] equals identity (and can be omitted).

    Syntax:

            normalize_diff(D, hD)

###### `discretization`:
    set of methods for discretizing the delay system
    
- discretize_ddae(E: npt.NDArray, A: npt.NDArray, hA: npt.NDArray, discretization: int, s0: complex=0j, method: str="cheb") -> tuple[npt.NDArray, npt.NDArray]:

    DDAE of form:

        E x'(t) = A[0] x(t) + A[1] x(t-hA[1]) + .. + A[m] x(t-hA[m]),      (1)

    is discretized into DAE of form:

        E x'(t) = A x(t).
    
    Syntax:

        discretize_ddae(E, A, hA)

###### `closed-loop`:

creates a TDS object that represents the closed-loop interconnection of the provided plant and controller.

dependencies:
- `.composition: concatenate_2x2_by_delays`
- `.compress: compress_matrices_delays, compress_bool_matrices_delays`

sub-functions:
- `controller_reprezentation`: creates an empty controller representation 


###### `composition`: 

consists of a set of functions for TDS composition

sub-functions:
- `concatenate_2x2_by_delays`: Concatenates system into compact form respecting delay vectors

    Assumes system is defined as

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

        E* = [E, 0]
             [0, 0]
    right hand-side array:
    
        A*[:,:,:n] = [A, 0]  
                     [0, 0]
        A*[:,:,n:n+m] = [0, B]  
                        [0, 0]
        A*[:,:,n+m:n+m+p] = [0, 0]  
                            [C, 0]
        A*[:,:,n+m+p:] = [0, 0]
                         [0, D]

- `interconnect`: Creates an interconnected system

    Args: \
        tds1 (TDS): system 1    \
        tds2 (TDS): system 2    \
        y1_indices (list): list of indices (outputs of system 1), if not defined, [0] is assumed    \
        u2_indices (list): list of indices (inputs of system 2), if not defined, [0] is assumed     \
        y2_indices (list): list of indices (outputs of system 2), if not defined, [0] is assumed    \
        u1_indices (list): list of indices (inputs of system 1), if not defined, [0] is assumed     

        **kwargs:
            compress (bool): perform compression of resulting system, default
                True

     Assume we have two systems:

        E1 dx1dt = SUM A1[i] x1(t-hA1[i]) + SUM B1[j] u1(t-hB1[j])
              y1 = SUM C1[k] x1(t-hC1[k]) + SUM D1[l] u1(t-hD1[l])

        E2 dx2dt = SUM A2[i] x2(t-hA2[i]) + SUM B2[j] u2(t-hB2[j])
              y2 = SUM C2[k] x2(t-hC2[k]) + SUM D2[l] u2(t-hD2[l])
        
    And interconnection defined via indices mapping, then the final system can
    be discribed via TODO


##### `compress`:

Set of functions for representing compressions i.e. obtaining a minimally sorted representation of the tds

sub-functions

- `compress_matrices_delays` compresses the matrices-delays representation. Removes delay duplicates, sorts the delays into ascending order and removes the matrices close to zero
- `compress_bool_matrices_delays` compresses boolean matrices - delays representation

    removes delay duplicates, and sorts delays into ascendinging

- `sort_matrices_delays` sorts delays into ascending order, i.e. changes the representation $ A_0 x(t-hA_0) + \ldots + A_{mA} x(t-hA_{mA})$ into the representation $A^*, hA^*$

- `compress_ddae` Not implemented yet


##### `delay_difference_equation` 

set of functions for obtaining and manipulating the delay difference equation

sub-functions:

- `ddae_to_diff`: converts the ddae into a delay difference equation
- `ndde_to_diff`: converts the ndde to delay-difference equation
- `_normalize_diff`: normalizes the delay-difference equation
- `normalize_diff`: normalizes the delay difference equation such that the leading zero delay is identity and can be omitted


##### `discretization`

discretizes the DDAE into a DAE (no checks performed)

sub-functions:

- `_discretize`: discretizes the DDAE into a DAE i.e.
A DDAE of form:

        E x'(t) = A[0] x(t) + A[1] x(t-hA[1]) + .. + A[m] x(t-hA[m]),      (1)

    is discretized into DAE of form:

        E x'(t) = A x(t).    

- `discretize_ddae`: discretizes the DDAE into a DAE
    DDAE of form:

        E x'(t) = A[0] x(t) + A[1] x(t-hA[1]) + .. + A[m] x(t-hA[m]),      (1)

    is discretized into DAE of form:

        E x'(t) = A x(t).        
    
- `discretize`: discretizes the RDDE, NDDE or DDAE into a DAE


##### `plot`

- `eigenvalues`:
    Set of eigenvalue plotting functions

##### `stability`

- `bounds`: 
    enforce the lowerbound and the upperbound

    - `lower_bound`: Calculates lower bound

            lower_bound(x, epsilon, gamma)

    - `upper_bound`: Calculates upper bound

            upper_bound(x, epsilon, gamma)

- `discretization_heuristic`: 
    function to compute N
- `characteristic_roots`:
    set of functionalities connected to computation of characteristic roots of a DDAE
- `compute_n_rect`:
- `gamma_r`:
    Computation of gamma(r, DIFF)
    -----------------------------

    DIFF - delay difference equation represented via matrices (3D array)
    and delays (1D array)
    computation of gamma(r,tds)

    sub-functions:

    - `func`: 
        Calculates value and jacobian of the vector function F(x)

        vector x, shape=(4*ndiff + 2 + (m-1), ):

            x = [Re(v), Im(v), Re(u), Im(u), Re(lambda), Im(s), th]
        
        function F(x):

            M * v - s * v                                                   = 0
            u^{H} * M  - s * u^{H}                                          = 0
            u^{H} * v  - 1                                                  = 0
            v0^{H} * v - 1                                                  = 0
            Im(conj(lambda)*(u^{H}*DD{k}*v)*exp(-r*hDD(k))*exp(1j*theta(k)) = 0 for k = 1,...,m
            
        with th = [0; th_v], v0 a normalization vector and
            
            M = DD[0]*exp(-r*hDD[0])*exp(1j*th[1]) + ... + DD[m]*exp(-r*hDD[m])*exp(1j*th[m])
            
        using fsolve (#optim. variables: 4*ndiff + 2 + (m-1), #constraints: 4*ndiff + 2 + (m-1) ).

        Returns:
            tuple containing

            - y (array): 1D array representing function F evaluated at x
            - jac (array): jacobian of F evaluated at x
        
        Notes:
            1. shape of `x` (n_opt, ), n_opt = 4*ndiff + 2 + (m-1)
            1. shape of `jac` (2*(2*n_diff+2) + n_opt, 2*(2*n_diff+1) + n_opt),
                n_diff ... dimension of the state vector of the delay-difference equation

    - `gamma_normalized_diff`: `gamma_r, gamma_info = gamma_normalized_diff(DD, hDD, r, **kwargs)`
        
        Computes gamma(r) of the normalized delay difference equation
    
        Normalized delay difference equation takes form
            0 = x(t) + DD[0] * x(t-hDD[0]) + ... + DD[m-1]*x(t-hDD[m-1]),      (1)
        where m == len(DD) == len(hDD).

        The gamma(r) of (1) is given by the following optimization problem:
            find maximum theta from [0, 2*pi)^m of expression:
                rho( SUM for all k DD[k]*exp(-r*hDD[k])*exp(1j*theta[k]) )     (2)
            where rho(.) is spectral radius of its matrix argument.

        Args:
            DD (array): coefficient matrices packed into 3D array shaped (n,n,m),
                note that coefficients for x(t) are assumed to be identity matrix
                and therefore omitted (see functions for converting DDAE to delay
                difference equation and normalizing)
            hDD (array): delays represented by 1D array shaped (m,), note that delay
                0 is omitted
            r (float): point from complex plane

            kwargs:
                n_theta (int): theta discretization, has to be > 0, default 10
                correction (bool): if correction is applied, default True
                scipy_root_method (str): scipy.optimize.root method, default 'lm',
                    i.e. Levenberg-Marquardt algorithm, note: carefull, not all
                    methods attempt to solve problem
                scipy_root_tol (float): Tolerance for termination. For detailed
                    control, use `scipy_root_options`, default None
                scipy_root_callback (function): Optional callback function. It is
                    called on every iteration as `callback(x, f)` where x is the
                    current solution and f the corresponding residual. For all
                    methods but 'hybr' and 'lm'.
                scipy_root_options (dict): a dictionary of solver options (method),
                    default None
        
        Returns:
            tuple containing:
                
                - gamma (float): quantity gamma(r, DD, hDD)
                - info (GammaInfo): gamma metadata
        
        Notes:
            1. for all kwargs starting with 'scipy_*' check the following documentation
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html

    - `gamma_diff`: `gamma_diff(D, hD, r)`
        Computes gamma(r) of delay difference equation (DIFF)

        Delay difference equation takes form
            0 = D[0]*x(t) + D[1] * x(t-hD[1]) + ... + DD[m-1]*x(t-hDD[m-1]),   (1)
        where m == len(DD) == len(hDD).

        quantity gamma(r; D, hD) is then:
            1. gamma(r; D, hD) = 0 IF number of delays (vector hD) is less then 2
            2. obtained via predictor corrector approach, i.e.
                2a. normalize DIFF (multiply equation (1) by inverse of D[0]) and
                    omit first delay = 0 and first normalized matrix = identity
                2b. call `gamma_diff_normalized`
        
        Args:
            D (array): coefficient matrices packed into 3D array shaped (n,n,m)
            hDD (array): delays represented by 1D array shaped (m,)
            r (float): point from complex plane
            **kwargs: kwargs passed into `gamma_diff_normalized` function
        
        Returns:
            tuple containing:
                
                - gamma (float): quantity gamma(r, D, hD)
                - info (GammaInfo): gamma metadata
        
        Notes:
            1. if compressed version of DIFF contains 2 or more delays,
                invertibility of D[0] is assumed.
            2. DIFF representation (D, hD) can be emtpy, result will be
                gamma(r; D, hD) = 0.0. Test for emptyness is hD.size == 0.
            3. for r = 0.0, quantity gamma(r; D, hD) DOES NOT depend on the delays,
                see implementation of `gamma_diff_normalized`

        


- `newton`:
    newton method for increasing the precision of roots
- `spectral_abscissa`:

##### `stabopt`:

- `stabopt`
- `gradient_sa`
- `gradient_gamma0`
- `gradient_CD`
- `fg_stab`
- `fg_gamma0`


##### `base`
Abstract parent class for RDDE, NDDE and DDAE.
Contains the fields:
- mA: number of state delays
- A: state matrices
- hA: state delays
- n: system order

##### `controller`

set of high-level API functions for creating controllers

dependencies:
- `composition`: `concatenate_2x2_by_delays`
- `compress`: `compress_matrices_delays`
- `closed_loop`: `controller representation`
    Assume we have two systems:

        E1 dx1dt = SUM A1[i] x1(t-hA1[i]) + SUM B1[j] u1(t-hB1[j])
              y1 = SUM C1[k] x1(t-hC1[k]) + SUM D1[l] u1(t-hD1[l])

        E2 dx2dt = SUM A2[i] x2(t-hA2[i]) + SUM B2[j] u2(t-hB2[j])
              y2 = SUM C2[k] x2(t-hC2[k]) + SUM D2[l] u2(t-hD2[l])
        
    And interconnection defined via indices mapping, then the final system can
    be discribed via TODO


    x* = [x1, u1, y1, x2, u2, y2]

sub-functions:

- `ClosedLoop` controller representation
- `interconnect` creates an interconnected system

    Args:
    
        tds1 (TDS): system 1
        tds2 (TDS): system 2
        y1_indices (list): list of indices (outputs of system 1), if not defined,
            [0] is assumed
        u2_indices (list): list of indices (inputs of system 2), if not defined,
            [0] is assumed
        y2_indices (list): list of indices (outputs of system 2), if not defined,
            [0] is assumed
        u1_indices (list): list of indices (inputs of system 1), if not defined,
            [0] is assumed
        **kwargs:
            compress (bool): perform compression of resulting system, default
                True

    Assume we have two systems:

        E1 dx1dt = SUM A1[i] x1(t-hA1[i]) + SUM B1[j] u1(t-hB1[j])
              y1 = SUM C1[k] x1(t-hC1[k]) + SUM D1[l] u1(t-hD1[l])

        E2 dx2dt = SUM A2[i] x2(t-hA2[i]) + SUM B2[j] u2(t-hB2[j])
              y2 = SUM C2[k] x2(t-hC2[k]) + SUM D2[l] u2(t-hD2[l])
        
    And interconnection defined via indices mapping, then the final system can
    be discribed via TODO


    x* = [x1, u1, y1, x2, u2, y2]

                E1, 0, 0,  0, 0, 0
                0, 0, 0,  0, 0, 0
        E =     0, 0, 0,  0, 0, 0
                0, 0, 0, E2, 0, 0
                0, 0, 0,  0, 0, 0
                0, 0, 0,  0, 0, 0
    

                E1, 0, 0,  0, 0, 0
                0, 0, 0,  0, 0, 0
        E =     0, 0, 0,  0, 0, 0
                0, 0, 0, E2, 0, 0
                0, 0, 0,  0, 0, 0
                0, 0, 0,  0, 0, 0

- `create_static_controller` creates a static controller from the matrix of coefficients. Static controller is assumed to be of a form

        y = K*u,
    but is constructed as DDAE with all delay equal to 0.0 and matrices A, B, C
    empty, i.e.

        I dxdt = A*x + B*u
             y = C*x + K*u

    **currently missing the delays**
- `create_dynamic_controller` creates a dynamic controller from the state-space representations. The form is assumed to be

        I dxdt = A*x + B*u
             y = C*x + K*u
    **currently missing the delays**
- `interconnect2` creates an interconnected system ready for stabilization

    Args:
        tds1 (TDS): system 1 to be interconnected
        y1_indices (list): indicies of measurements
        u1_indices (list): indicies of controled inputs
        hA2 (array): controller delays, default None will assume delay vector
            to be [0.0]
        hB2
        hC2
        hD2

    Returns:

        interconnected system (DDAE)

    

- `interconnect3` creates a closed-loop representation
    Args:
    
        tds1 (TDS): system 1 to be interconnected
        y1_indices (list): indicies of measurements
        u1_indices (list): indicies of controled inputs
        hA2 (array): controller delays, default None will assume delay vector
            to be [0.0]
        hB2
        hC2
        hD2

    Returns:
        interconnected system (DDAE)

- `create_closed_loop` creates a new TDS object representing closed-loop interconnection of the provided plant and controller
    
    Args:

        plant
        controller

        **kwargs

    Returns
        tuple containing:

            - closed_loop
            - closed_loop_metadata


##### `closed_loop`

##### `dae`
Differential algebraic equation

##### `ddae`
Class defining the DDAE impementation.



##### `gamma`
implementation of the function tds_gamma_r

##### `ndde`
neutral delay differential equation

##### `rdde`
retarded delay differential equation

##### `roots`
implementation of the function `tds_roots`. The function computes the characteristic roots of a given delay-differential equation or time-delay system in a specified right half-plane or rectangular 


##### `spectral_abscissa`

##### `zeros`
functionalities for computing the transmission zeros


### `Examples`

#### `example01`: RDDE

Example 2.1 from the TDS-CONTROL manual
We will analyze the exponential stability of the following RDDE from [1,
Section 6.1]: 

    x'(t) = A0 x(t) + A1 x(t-1)

[1] Verheyden K., Luzyanina T., and Roose D. (2008). Efficient
    computation of characteristic roots of delay differential equations
    using LMS methods. Journal of Computational and Applied Mathematics,
    214(1), pp. 209–226.  

#### `example02`: 

Example 2.6 from the TDS-CONTROL manual
We will analyze the stability of the following NDDE from page 16, equation (2.20)

    x'(t) = 1/4 x(t) - 1/3 x(t-tau1) + 3/4 x'(t-tau1) - 1/2 x' (t-tau2)


Compute the roots for the above NDDE for 

    tau1 = 1, tau2 = 2

#### `example_strong_sa02`:

Example 2.7 from the TDS-CONTROL manual
Compute the strong spectral abscissa for the NDDE from page 16, equation (2.20)

    x'(t) = 1/4 x(t) - 1/3 x(t-tau_1) + 3/4 x'(t-tau1) - 1/2 x' (t-tau2)

Compute the roots for the above NDDE for 

    tau1 = 1, tau2 = 2


#### `example_
