Project structure and hierarchy
-------------------------------
-------------------------------

TDSpy package
-------------------------------

* `docs`
    * `coming_from_matlab.md`
* `examples`
* `src`
    * `tdspy`
        * `common`
            * `init`
            * `closed_loop`
            * `composition`
            * `compress`
            * `delay_difference_equation`
            * `discretization`
        * `plot`
            * `init`
            * `eigenvalues`
        * `stability`
            * `init`
            * `bounds`
            * `characteristic_roots`
            * `discretization_heuristic`
            * `gamma_r`
            * `newton`
            * `spectral_abscissa`
        * `stabopt`
            * `init`
        * `init` 
        * `base`         Abstract parent class for RDDE, NDDE and DDAE
        * `controller` 
        * `dae`          Differential algebraic equation
        * `ddae`     DDAE impementation
        * `gamma`    implementation of the function tds_gamma_r
        * `ndde`     neutral delay differential equation
        * `rdde`     retarded delay differential equation
        * `roots`    implementation of the function tds_roots
        * `zeros`    functionalities for computing the transmission zeros
        * `spectral_abscissa`
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
    set of functionalities connected to computation of characteristic roots of a DDAE. 
- `compute_n_rect`: 
- `gamma_r`



.
├───src
    ├───common
    ├───plot
    ├───stability
    └───stabopt
├───examples <- examples for usage
├───docs <- documentation
├───test <- pytest
│
│
├───README.md
├───.gitignore
├───requirements.txt
├───pyproject.toml
└─── ...

