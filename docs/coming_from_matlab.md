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

- `delay_difference_equation`:
    Set of functions for obtaining and manipulating the delay-difference equations
- `discretization`:
    set of methods for discretizing the delay system

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

- `interconnect`:


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