Controller Design
====================

`tdcpy` can be used for designing the following classes of controllers for time-delay systems:

- Static feedback controllers
- Dynamic controllers
- Delayed feedback controllers

The software natively supports the design of controllers for systems described by delay-differential algebraic equations (DDAEs) of the form

.. math::
    :label: eq_plant
    
    E \dot{x}(t) = \sum_{i=0}^{m_A} A_i x(t - \tau_i) + \sum_{i=0}^{m_B} B_i u(t - \tau_i)
    
    y(t) = \sum_{i=0}^{m_C} C_i x(t - \tau_i) + \sum_{i=0}^{m_D} D_i u(t - \tau_i)

where :math:`(A_i, B_i, C_i, D_i)` are the system matrices corresponding to their respective delays.
Both retarded and neutral systems can be easily converted to DDAEs, which is why the above model is rather convenient.

In `tdcpy`, the controller, whether static, dynamic, or delayed, can generally be defined as a DDAE of the form:

.. math::
    :label: eq_controller
    
    \dot{x}_c(t) = \sum_{i=0}^{m_{A_c}} A_{c_i} x_c(t - \tau_i) + \sum_{i=0}^{m_{B_c}} B_{c_i} y(t - \tau_i)

    u(t) = \sum_{i=0}^{m_{C_c}} C_{c_i} x_c(t - \tau_i) + \sum_{i=0}^{m_{D_c}} D_{c_i} y(t - \tau_i)

where :math:`x_c(t) \in \mathbb{R}^{n_c}` is the controller state, :math:`n_c` denoting the order of the controller, and :math:`y(t)` is the system output.

In the packed representation, the controller gain matrix can be viewed as:

.. math::

    K = \left[ \begin{array}{c | c}
     A_C    &   B_C \\
     \hline
     C_C    &   D_C \\
    \end{array} \right] 

The interconnection of the plant and the controller can be visualized in the following block diagram:

    .. code-block:: text
                
                                     ___________________
                        u1[-1]      |                   |  y1[-1]    
                    --------------->|                   |---------------->
                u1[0],...,u1[nu-2]  |      SYSTEM       |  y1[0],...,y[ny-2]
                             ------>|                   |-------
                            |       |___________________|       |
                            |                                   |
                            |        ___________________        |
        y2[0],...,y2[ny-2]  |       |                   |       | u2[0],...,u1[nu-2]
                             -------|                   |<------
                        y2[-1]      |    CONTROLLER     |  u2[-1]
                    <---------------|                   |<---------------
                                    |___________________|


We distinguish between `TDS-Control` and `tdcpy` in how the inputs and outputs are defined.
`tdcpy`, the user does not specifically differentiate between the matrices :math:`C_1, C_2` corresponding to the control outputs :math:`y(t)` (for feedback) and the system outputs :math:`z(t)` separately,
or between the matrices :math:`B_1, B_2` corresponding to the respective control inputs :math:`u(t)` (for actuation) and the exogenous inputs :math:`w(t)`.
Instead, the columns of the input matrix :math:`B` are stacked horizontally to include both the control inputs and the exogenous inputs, with the user specifies
the indices corresponding to the control inputs `u_indices`. The remaining indices are automatically selected as `w_indices`. 
The same case goes for the outputs, with `y_indices` and the `z_indices` the indices of the control outputs and performance outputs, respectively.
We wish for the reader to keep the above picture in mind, as it shall be useful in the subsequent controller design.

As a preliminary step to the controller design, the user defines the `DDAE` as

.. code-block:: python

    from tdcpy import DDAE
    ddae = DDAE(E, A, hA, B, hB, C, hC, D, hD)

Alternately, if the system is defined as an `NDDE`, the user simply converts the `NDDE` to a `DDAE` using the `to_ddae` method.

Defining a controller structure
--------------------------------

For defining the controller, the user may proceed as follows:

1. **Static output feedback controller:**

A controller with static output feedback is defined as follows:

.. math::

    u(t) = K y(t)

The simplest way to define such a controller structure is:

.. code-block:: python
    
    K = np.zeros([nu, ny, 1])       # dimensions (nu, ny, m_D+1)

Here, the controller is of order :math:`n_c=0` with no feedback delays i.e. `m_D=0`. 
Comparing with :eq:`eq_controller`, the matrices :math:`A_{c_i}, B_{c_i}, C_{c_i}` are zero for all :math:`i`, 
with :math:`D_{c_0} = K`. The function :func:`tdcpy.ClosedLoop` is then used to interconnect the plant and the controller to form the closed-loop system.

.. code-block:: python

    from tdcpy import ClosedLoop
    closed_loop = ClosedLoop(ddae,order,y_indices,u_indices,K)

.. note::
    :collapsible:

    A **second** approach to define a static output feedback controller is to use the :func:`create_static_controller` function from the 
    :mod:`tdcpy.controller` module. Using this method, the software creates the controller directly as a `DDAE` object. The syntax is as follows:

    .. code-block:: python

        from tdcpy.controller import create_static_controller
        K = np.zeros([nu, ny])
        controller = create_static_controller(K=K)

    the function returns a `DDAE` object with the fields `controller.A`, `controller.B`, `controller.C` as zero matrices, and `controller.D` containing the static gain `K`.
    The interconnection of the plant and the controller is be performed using the `interconnect` function from the :mod:`tdcpy.controller` module:

    .. code-block:: python

        from tdcpy.controller import interconnect
        closed_loop = interconnect(ddae, controller, y_indices, u_indices)


2. **Dynamic controller:**

A dynamic controller of order :math:`n_c > 0` can be defined as follows

.. math::
    
    \dot{x}_c(t) = A_c x_c(t) + B_c y(t) \\
    
    u(t) = C_c x_c(t) + D_c y(t)

Similar to the static case, once the matrices :math:`A_c, B_c, C_c, D_c` are defined, the controller can be created using:

.. code-block:: python
    
    from tdcpy.controller import create_dynamic_controller
    controller = create_dynamic_controller(Ac=Ac, Bc=Bc, Cc=Cc, Dc=Dc)

The function returns a `DDAE` object with the fields `controller.A`, `controller.B`, `controller.C`, `controller.D` containing the respective matrices of the dynamic controller 
and the interconnection of the plant and the controller is be performed using the `interconnect` function.

3. **Delayed and dynamic feedback controller:**

In a similar way, a delayed and dynamic feedback controller can be defined as per controller equation :eq:`eq_controller`.

In `tdcpy`, a neat way of creating such a controller structure for dynamic controllers is by using the function `concatenatw_2x2_by_delays` from the :mod:`tdcpy.common.composition` module. 
The function creates a DDAE representation of the controller matrices by concatenating the respective matrices :math:`A_c, B_c, C_c, D_c` with the appropriate delays, 
and internally adding appropriate slack variables where necessary. This enables an equivalent controller representation in the form:

.. math::
    
    \tilde{u}(t) = K \tilde{y}(t)

with :math:`K = \begin{bmatrix} A_c & B_c \\\\ C_c & D_c \end{bmatrix}`.
          
The above structure is later used for creating the subsequent closed-loop using the :func:`tdcpy.ClosedLoop` function.

.. code-block:: python
    
    from tdcpy.closed_loop import ClosedLoop, concatenate_2x2_by_delays
    from tdcpy.controller import create_dynamic_controller
    from tdcpy import DDAE

    Ac, hAc = np.zeros((nc, nc)), np.zeros(nAc)
    Bc, hBc = np.zeros((nc, ny)), np.zeros(nBc)
    Cc, hCc = np.zeros((nu, nc)), np.zeros(nCc)
    Dc, hDc = np.zeros((nu, ny)), np.zeros(nDc)

    cont = DDAE(A=Ac,  B=Bc, C=Cc, D=Dc, hA=hAc, hB=hBc, hC=hCc, hD=hDc)        # create a dynamic controller as a DDAE object
    E, K, hK = concatenate_2x2_by_delays(cont.E, cont.A, cont.B, cont.C, cont.D, cont.hA, cont.hB, cont.hC, cont.hD)    # concatenate the controller matrices into a single matrix K
    closed_loop = ClosedLoop(ddae, nc=0, y1_indices, u1_indices, K, hK)           # form the closed-loop

The `cont` object is now a `DDAE` with the fields `cont.A`, `cont.B`, `cont.C`, `cont.D` containing the respective matrices of the dynamic controller.
The argument `nc=0` may seem redundant in this setting, since the controller has been defined already. However, it can be helpful when incorporated in high-level routines when the controller order is 
increased iteratively. An alternate way of forming the closed-loop is by using the `interconnect` function.


**Summary**

==================================================     ===============================================================================
    Command                                                     Description
==================================================     ===============================================================================
:func:`ClosedLoop`                                      Interconnect the plant and the controller to form the closed-loop system
:func:`create_static_controller`                        Create a static output feedback controller as a DDAE object
:func:`interconnect`                                    Interconnect the plant and the controller to form the closed-loop system  
:func:`create_dynamic_controller`                       Design a stabilizing controller using the BFGS optimization method
:func:`concatenate_2x2_by_delays`                       Concatenate the controller matrices into a single matrix K
==================================================     ===============================================================================

In what follows, we will see how the above controller structures can be used for designing stabilizing controllers.


Design of a stabilizing controllers for a retarded time-delay system
--------------------------------------------------------------------------------

We consider the interconnection of a retarded time-delay system defined by:

.. math::
    
    \dot{x}(t) = A_0 x(t) + \sum_{i=1}^{m} A_i x(t - \tau_i) + B u(t)

    y(t) = C x(t)

with a static output feedback controller of the form:

.. math::
    
    u(t) = K y(t)


The above interconnection yields a closed-loop defined as:

.. math::
    
    \dot{x}(t) = A_0 x(t) + \sum_{i=1}^{m} A_i x(t - \tau_i) + B K C x(t)

The characteristic equation of the closed-loop system reads as:

.. math::
    
    \det(\lambda I - A_0 - \sum_{i=1}^{m} A_i e^{-\lambda \tau_i} - B K C) = 0

and its spectral abscissa:

.. math::
    
    \alpha = \sup \{ \text{Re}(\lambda) : \det(\lambda I - A_0 - \sum_{i=1}^{m} A_i e^{-\lambda \tau_i} - B K C) = 0 \}

For the above system to be stable, we require that :math:`\alpha < 0`. 
The stabilization objective is therefore to find controller parameters :math:`K` such that the spectral abscissa of the closed-loop system is negative.

For desiging a stabilizing controller, we used the `design_bfgs` function from the :mod:`tdcpy.stabopt` module, which utilizes the `L-BFGS` solver from the `scipy.optimize` library.
The below example demonstrates the design of a stabilizing controller for a retarded time-delay system.

**Summary**

==================================================     ===============================================================================================================================
    Command                                                     Description
==================================================     ===============================================================================================================================
:func:`design_bfgs`                                     Low-level function for stabilization of retarded systems
:func:`func_sa`                                         Function to compute the objective function and the gradient of the spectral abscissa with respect to the controller parameters
:func:`minimize_spectral_abscissa`                      High-level API for designing a stabilizing controller
==================================================     ===============================================================================================================================



.. admonition:: Example: Design of a stabilizing controller for a retarded time-delay system
    :class: example

    We consider the retarded system defined by the system matrices:

    .. code-block:: python

        from tdcpy.stabopt.controller_bfgs import minimize_spectral_abscissa
        from tdcpy.ddae import DDAE
        A0 = np.array([[-1., 0.], [0., -2.]])
        A1 = np.array([[1., 0.], [0., 1.]])
        A = np.stack([A0, A1], axis=2)
        hA = np.array([0., 1.])
        Bu = np.array([ [-0.1],[-0.2]    ])
        B = np.stack([Bu],axis=2)
        hB = np.array([0.5])
        C = np.array(np.eye(2))
        C = np.stack([C],axis=2)
        hC = np.array([0])
        D = np.zeros(shape=(2,1,1), dtype=float)    # must be defined, otherwise error!
        hD = np.array([0.])
        ddae = DDAE(A=A,hA=hA,B=B,hB=hB,C=C,hC=hC,D=D,hD=hD)

        sa = tds.spectral_abscissa(ddae)
        print(f"Spectral abscissa of the open-loop system: {sa:.4f}")

        sol = minimize_spectral_abscissa(ddae, order=0, method="L-BFGS-B", options={"disp": True}, type = "barrier")
        print(f"Optimal controller parameters: {sol.x}, objective function value: {sol.fun}, success: {sol.success}, message: {sol.message}")

The function `minimize_spectral_abscissa` returns an optimization result `sol` which is a tuple containing the result from the optimization.



