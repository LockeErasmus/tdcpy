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
            * `init`
            * `closed_loop`
            * `composition`
            * `compress`
            * `delay_difference_equation`
            * `discretization`
        * [`plot`](#plot)
            * `init`
            * `eigenvalues`
        * [`stability`](#stability)
            * `init`
            * `bounds`
            * `characteristic_roots`
            * `discretization_heuristic`
            * `gamma_r`
            * `newton`
            * `spectral_abscissa`
        * [`stabopt`](#stabopt)
            * `init`
        * `init` 
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
- `closed-loop`:
    creates a TDS object that represents the closed-loop interconnection of the provided plant and controller.
- `composition`: 

- `compress`:
    `compress` sorts the delay values within each field and removes duplicate delays in each field by summing the system matrices associated with the same delay

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