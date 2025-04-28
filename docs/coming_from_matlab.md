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
    computation of gamma(r,tds)
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

#### `example01`:

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
