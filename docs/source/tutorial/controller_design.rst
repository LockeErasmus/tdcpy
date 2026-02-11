Stabilization and Controller Design
============================================


We consider the stabilization of time-delay systems described by the DDAE:

.. math::
    
    E \dot{x}(t) = P_0 x(t) + \sum_{i=1}^{m} P_i x(t - \tau_i)

using a controller of the form

.. math::
    
    \dot{x}_c(t) = \sum_{i=0}^{m_{A_c}} A_{c_i} x_c(t - \tau_i) + \sum_{i=0}^{m_{B_c}} B_{c_i} y(t - \tau_i)  \\
    
         u(t) = \sum_{i=0}^{m_{C_c}} C_{c_i} x_c(t - \tau_i) + \sum_{i=0}^{m_{D_c}} D_{c_i} y(t - \tau_i)

where :math:`y(t) = C x(t)` is the system output, and :math:`p \in \mathbb{R}^{n_p}` are the controller parameters.

By defining an augmented state vector :math:`\tilde{x}(t)`, the resulting closed-loop system can be written as

.. math::
    
    E \dot{x}(t) = \underbrace{(P_0 + B K_0(p) C)}_{A_0(p)} x(t) + \sum_{i=1}^{m} \underbrace{(P_i + B K_i(p) C)}_{A_i(p)} x(t - \tau_i)

where the system matrices :math:`P_0, P_1, \ldots, P_m` are the plant matrices and :math:`K_i` the respective controller gains corresponding to delays :math:`\tau_i`, with :math:`p \in \mathbb{R}^{n_p}`.
The stabilization objective is to find controller parameters :math:`p` such that the closed-loop system is stable, i.e., all characteristic roots have negative real part.

In `TDSpy`, the controller is by default a :class:`tdspy.ddae` object, which allows for a more generic controller structure. 
The above formulation allows for the design of controllers of the following types:

- Static feedback controllers
- Dynamic controllers
- Delayed feedback controllers


Pre-checks
-------------

Before proceeding with the stabilization procedure, it is important to determine whether the system is of retarded type or neutral type, as this will determine the choice of stabilization approach.

.. code-block:: python

    >>> is_neutral = ddae.is_essentially_neutral


Case 1: Retarded system
------------------------

If the system is of **retarded** type, then the stabilization problem can be solved by minimizing the spectral abscissa of the closed-loop system using a gradient-based optimization algorithm, for example the BFGS quasi-Newton method.
The function ``design_bfgs`` from the :mod:`tdspy.stabopt.controller_bfgs` module can be used for this purpose.

.. code-block:: python

    >>> sol = design_bfgs(E, P, hP, K0, hK, B, C)


Case 2: Neutral system
-------------------------

If the system is of **neutral** type, then the stabilization problem is more challenging, as the spectral abscissa may be sensitive to small perturbations.
In this case, we can either minimize the strong spectral abscissa of the closed-loop system, or we can solve a constrained optimization problem where the objective is to minimize the spectral abscissa.


**Pre-requisites**
The stabilization procedure relies on the following assumptions:

1. The system is stabilizable with the selected controller structure. This can be checked using the following command:

    .. code-block:: python

        # extract controller parameters affecting the delay-difference equation
        from tdspy.stabopt.utils import diff_dependency_mask
        r = diff_dependency_mask(Kmask, uE=uE, vE=vE, B=B, C=C)
        
        if not np.any(r):
            raise ValueError("The delay difference equation is independent of the controller parameters, no feasible point exists.")    

Note that the above condition ensures that the selected controller structure can influence the neutral dynamics. 
If the above condition is not satisfied, then we check if the spectrum of the delay-difference equation is stable. If yes, then proceed to minimize the spectral abscissa. 
If not, the system cannot be stabilized with the selected controller structure.

2. The gradient exists and is finite. This can be checked by computing :math:`\gamma_0` and :math:`C_D` at the initial controller parameters and checking if the value is finite.

    .. code-block:: python

        from tdspy.stability.gamma_r import gamma_diff
        g0, gammaInfo = gamma_diff(DD, hDD,r=0)
        if not np.isfinite(g0):
            raise ValueError("The gradient is not finite at the initial controller parameters, please choose a different initial point.")


If the above conditions are satisfied, then we can proceed with the stabilization procedure, which consists of two :


Option 1: Minimization of the strong spectral abscissa
--------------------------------------------------------

Under the control paradigm, to achieve exponential stability of the closed-loop, it is required to find controller parameters :math:`p` 
such that the spectral abscissa :math:`\alpha` of the closed-loop is strictly negative.  
Furthermore since the closed-loop bears neutral dynamics, the spectral abscissa may be sensitive to small perturbations, in which case, we consider the strong spectral abscissa :math:`C_D`.

The requirement for attaining strong stability of the closed-loop can thus be formulated as a constrained optimization problem of the form:

.. math::

    \min_{p} \quad & C(\tau;p) \\

where :math:`C` denotes the strong spectral abscissa of the closed-loop system, with :math:`C = \max(\alpha, C_D)`.




Option 2: Stabilization via constrained optimization
-------------------------------------------------------


The optimization problem can be stated as:

.. math::

    \min_{p} \quad & \alpha(\tau;p) \\
    \text{subject to} \quad & \gamma_0(p) < \gamma,

where :math:`\gamma_0` denotes the spectral radius of the difference operator of the neutral system, and :math:`\gamma < 1` is a prescribed upper bound.


Solution approach
~~~~~~~~~~~~~~~~~

The above constrained optimization problem is converted into an unconstrained optimization problem using a penalty method, 
for example the logarithmic barrier method. 
The first step consists of finding a feasible point :math:`p_0` such that :math:`\gamma_0(p_0) < \gamma`.

Once a feasible point has been found, one can solve the subsequent unconstrained optimization problem which takes the form:

.. math::
    \min_{p} \quad \alpha(\tau;p) - r \log(\gamma - \gamma_0(p)),

where :math:`r > 0` is the barrier parameter.

The resulting unconstrained optimization problem is then solved using a gradient-based optimization algorithm, where the gradients


Algorithm 1: logarithmic barrier method for stabilization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _algorithm1:

**Optimization problem:**

.. math::
    \min_{p} \quad \alpha(\tau;p) - r \log(\gamma - \gamma_0(p)),

where :math:`r > 0` is the barrier parameter.


**Step 1: Determine whether the time-delay system is of neutral type or retarded type.**

This can be checked directly by using the command ``is_essentially_neutral`` from ``ddae`` module as follows:

    >>> is_neutral = ddae.is_essentially_neutral

The function finds the delay-difference equation and then checks whether the characteristic matrix of the system depends on :math:`s`.

The associated delay-difference equation can be written as

.. math::
    D_0 x_2 (t) + \sum_{i=1}^N D_i(p) x_2 (t-\tau_i) = 0

with :math:`D_0 = U^T A_0 V`, :math:`D_i = U^T (A_i + B K_i(p) C) V`.

In the normalized form, this can be written as

.. math::
    I x_2 (t) + \sum_{i=1}^N H_i(p) x_2 (t-\tau_i) = 0
with :math:`H_i = D_0^{-1} D_i(p)`.

.. 1. Find the left and right null vectors of E: :math:`U` and :math:`V`.

..     >>> U = linalg.null_space(E.T, rcond=rcond)
..     >>> V = linalg.null_space(E, rcond=rcond)

.. 2. Right multiply by :math:`V` and left multiply by :math:`U^{T}` 
.. to obtain the reduced system matrices:
    
..     >>> A0_bar = U.T @ A0 @ V
..     >>> Ai_bar = [U.T @ Ai @ V for Ai in A[:,:,1:]]

.. 3. obtain the associated delay-difference equation 
    
.. .. math::
..     D_0 x_2 (t) + \sum_{i=1}^N D_i x_2 (t-\tau_i) = 0

.. with :math:`D_0 = U^T A_0(p) V`, :math:`D_i = U^T A_i (p) V`

.. This can be obtained using the function `ddae_to_diff`

..     >>> D,hD = ddae_to_diff(E,A,hA)

.. Obtain also the normalized delay-difference equation

.. .. math::
..     I x_2 (t) + \sum_{i=2}^N H_i x_2 (t-\tau_i) = 0
.. with :math:`H_i = D_0^{-1} D_i`


..     >>> DD,hDD = normalize_diff(D,hD)

.. 4. Check if the characteristic matrix depends on :math:`s` as

.. .. math::

..     \Delta_D (s) = D_0 + \sum_{i=1}^N D_i e^{-s \tau_i}

.. If :math:`\det(\Delta_D(0)) = 0`, then the system is of neutral type, else it is of retarded type.

.. Alternately, this can be checked using the following command:

..     >>> if hD.shape[0] == 1 and hD[0] == 0.0:
..     >>>    # retarded type
..     >>> else:
..     >>>     # neutral type


**Step 2: Depending on the system type, proceed as follows:**

i. If ``is_essentially_retarded``, then solve the unconstrained optimization problem
    
**Optimization problem:**

.. math::

    \min_{p} \quad \alpha(\tau;p),

using a gradient-based optimization algorithm, using the function ``design_bfgs`` from the :mod:`tdspy.stabopt.controller_bfgs.design_bfgs` module.

ii. If ``is_essentially_neutral``, proceed to Step 3.


**Step 3: Find a feasible point** :math:`p_0` **such that** :math:`\gamma_0(p_0) < \gamma`

Check if the initial controller parameters :math:`p_0` satisfy the feasibility condition :math:`\gamma_0(p_0) < \gamma`.
If not, solve the optimization problem to obtain a feasible point :math:`p_0`.

.. math::
    
    p  \rightarrow \min_{p} \quad \gamma_0(p).

Refer :ref:`algorithm2` for details on the optimization problem and gradient computation.

**Step 4: Solve the unconstrained optimization problem**

.. math::

    \min_{p} \quad \alpha(\tau;p) - r \log(\gamma - \gamma_0(p)),

using a gradient-based optimization algorithm.



Algorithm 2: Finding a feasible point for neutral systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _algorithm2:

.. math::
    
    \min_{p} \quad \gamma_0(p).

with gradient

.. math::

    \frac{\partial \gamma_0(p)}{\partial p_k}  = \frac{1}{|\lambda|} \Re \left( \bar{\lambda} u^* \left( \frac{\partial H_1(p)}{\partial p_k} +
                         \sum_{i=2}^m \frac{\partial H_i}{\partial p_k} e^{j\theta_i}  \right) v \right),

where :math:`(\lambda, u, w, \theta)` are obtained from the function ``gamma_diff``.


**Step 1: Precomputations**

1. Extract the **original** delay difference equation (without controller)

.. math::
    D_{P_0} x(t) + \sum_{i=1}^m D_{P_i} x(t-\tau_i) = 0, \quad i = 0,1,\ldots,m


.. code-block:: python
    
    >>> DP, hDP = ddae_to_diff(E, P, hP)


2. Precompute the following **initial** quantities:

Obtain :math:`B_U = U^T B`, :math:`C_V = C V` with :math:`U` and :math:`V` the left and right null spaces of :math:`E`.
    
    >>> BU = U.T @ B
    >>> CV = C @ V
    

4. Now call the optimization solver with gradient function ``grad_gamma0`` from the :mod:`tdspy.stabopt.gradients` module to solve

    >>> grad_gamma0(p, DP, hDP, indices, hK, BD, CD)


**Step 2: Solve the unconstrained optimization problem**


.. code-block:: python

    >>> sol = optimize.minimize(
    >>>         grad_gamma0,
    >>>         x0,
    >>>         args=(DP, hDP, Kmask, hK, uE.T @ B, C @ vE),
    >>>         jac=True,
    >>>         method=kwargs.get("method", "L-BFGS-B"),
    >>>         options=options,
    >>>         callback=kwargs.get("callback", None)
    >>>       )

with ``grad_gamma0``

    >>> g0, grad = grad_gamma0(p, DP, hDP, Kmask, hK, BU, CV)

1. **Gradient computation - precomputation:**

i. Form the matrices of the DDE with controller parameters :math:`p`

.. math::

    D_i(p) = D_{P_i} + B_U K_i C_V, \quad i = 1,\ldots,m


.. code-block:: python

    >>> # set controller parameters
    >>> K = x.reshape((nu, ny, hK.shape[0]))
    >>> BK  = np.einsum('lm,mki->lki', BU, K)   # B @ K_i for all i
    >>> BKC = np.einsum('lki,kn->lni', BK, CV)  # (B @ K_i) @ C
    >>> D, hD = np.concatenate([DP, BKC], axis=2), np.concatenate([hDP, hK], axis=0)

ii. Obtain the normalized delay-difference equation

.. math::

    I x(t) + \sum_{i=1}^m H_i(p) x(t-\tau_i) = 0, \quad i = 1,\ldots,m

.. code-block:: python

    >>> DD, hDD = normalize_diff(D, hD)

iii. Compute the quantities :math:`(\lambda, u, v, \theta)` associated with :math:`\gamma_0(p)`

.. code-block:: python

    >>> g0, gammaInfo = gamma_normalized_diff(DD, hDD,r=0)
    >>> s, u, v, th = gammaInfo.s, gammaInfo.u, gammaInfo.v, gammaInfo.th

2. **Gradient computation - initialization:**

Initialize the gradient vector as zeros

.. code-block:: python

    >>> grad = np.zeros_like(p)

3. **Gradient computation - closed-form expression:**

.. math::

    \frac{\partial \gamma_0(p)}{\partial p_k}  = \frac{1}{|\lambda|} \Re \left( \bar{\lambda} u^* \left( \frac{\partial H_1(p)}{\partial p_k} +
                         \sum_{i=2}^m \frac{\partial H_i}{\partial p_k} e^{j\theta_i}  \right) v \right),

where :math:`(\lambda, u, w, \theta)` are obtained from the function ``gamma_diff``.


i. Computing the gradients :math:`\partial H_i(p)/\partial p_k`:

.. math::

    \frac{\partial H_i(p)}{\partial p_k} = - D_0^{-1} \left( \frac{\partial D_0}{\partial p_k} \right) H_i(p) + D_0^{-1} \left( \frac{\partial D_i(p)}{\partial p_k} \right)


ii. Computing the gradients :math:`\partial D_i(p)/\partial p_k`:

.. math::
    \frac{\partial D_i(p)}{\partial p_k} = B_U \left( \frac{\partial K_i(p)}{\partial p_k} \right) C_V

with :math:`\partial D_0/\partial p_k = 0`.

iii. Computing the closed-form expression for the gradient: :math:`\frac{\partial \gamma_0(p)}{\partial K}`

.. math::

    \frac{\partial \gamma_0(p)}{\partial K} = \frac{1}{|\lambda|} \Re \left(  \sum_{i=1}^m \bar{\lambda} \underbrace{(u^* D^{-1} B_U )^T}_{uDBU} \underbrace{(C_V v)^T}_{CVv} e^{j\theta_i} \right)

.. code-block:: python

    >>> uDBU = (np.conj(u).T @ linalg.solve(D[:,:,0], BU)).T # dimensions (nu,)
    >>> CVv = (CV @ v).T # dimensions (ny,)
    >>> for i in range(1, m+1):
    >>>     grad += np.real(np.conj(s) * uDBU * CVv * np.exp(1j*th[i-1])) / np.abs(s)
    >>> grad = grad.reshape(-1)
    


The following code snippet illustrates the stabilization of a neutral time-delay system using the above approach.







