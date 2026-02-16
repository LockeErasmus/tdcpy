Creating TDS Objects
==============================

.. contents:: Contents
   :local:


A time-delay system (TDS) is a dynamical system in which the evolution of the state depends not only on its current state but also on its past states.
Such systems are commonly modeled using delay-differential equations (DDEs) which are differential equations that include terms with delays.

Depending on the nature of the delays and their influence on the system dynamics, time-delay systems can be categorized into different classes, viz. retarded, neutral, and delay descriptor systems.
``TDSpy`` allows to model linear time-invariant (LTI) time-delay systems with multiple discrete delays, represented as delay-differential equations (DDEs) or as delay-differential-algebraic equations (DDAEs).

We consider the following classes of time-delay systems and their corresponding `TDSpy` objects:
    
=============================================== ============================
 **Type**                                           **TDSpy class**        
=============================================== ============================
 Retarded DDEs (RDDEs)                              :class:`tdspy.rdde`     
 Neutral DDEs (NDDEs)                               :class:`tdspy.ndde`     
 Delay-differential algebraic equations (DDAEs)     :class:`tdspy.ddae`     
=============================================== ============================

The :class:`tdspy.base` class provides the abstract parent class on which the above `rdde`, `ndde`, and `ddae` classes are based.
Each class comes with its own set of defined methods and properties, which are detailed in the :doc:`API reference <../reference/api_reference>` section of the documentation.

Besides, the software also handles quasi-polynomial representation of time-delay systems, as explained in the next section.

Retarded DDEs
---------------------

``TDSpy`` handles LTI retarded time-delay systems with multiple discrete delays, represented by delay-differential equations (DDEs) of the general form:

.. math::
    :label: eq_rdde

    \dot{x}(t) = \sum_{i=0}^{m_A} A_i x(t - h_i) + \sum_{j=1}^{m_B} B_j u(t - h_{B_j}) 

    y(t) = \sum_{i=0}^{m_C} C_i x(t - h_i) + \sum_{j=0}^{m_D} D_j u(t - h_{D_j})

where :math:`x(t) \in \mathbb{C}^n` is the state variable, :math:`A_0, \ldots, A_m` are constant matrices of size :math:`n \times n`, corresponding to delays :math:`h_1, \ldots, h_m` respectively.    
The matrices :math:`B_j` and :math:`C_i` correspond to the input and output matrices respectively, and the matrices :math:`D_j` correspond to the feedthrough elements.
We will skip the input and output terms for now, and will focus mainly on the system matrices as these define the spectral properties.


In ``TDSpy``, a retarded time-delay system can be created using the (high-level) :func:`tdspy.rdde` class.

.. admonition:: Example: Simple problem
    :class: example

    Consider a simple retarded time-delay system, described by the delay-differential equation:

    .. math::

        \dot{x}(t) = A_0 x(t) + A_1 x(t - h_1)

    where :math:`x(t)` is the state variable, :math:`A_0` and :math:`A_1` are constant matrices, and :math:`h_1` is the delay.

    We create a ``TDSpy`` object defining this system using the following code snippet:


    .. code-block:: python

        A0 = np.array([[1, 0], [0, 1]], dtype=float)
        A1 = np.array([[0, 1], [0.5, 0]], dtype=float)
        delays = np.array([0,0.5])
        rdde = tds.RDDE(A = [A0, A1], hA=delays)
  
In the above code snippet, the first delay is zero (corresponding to the current state) and the second delay is :math:`0.5`.
The software automatically compresses the delays and sorts the system matrices in ascending order of delays.
The above works similar to the `TDS-Control` function `tds_create('A',{A0,A1},'hA',[0,0.5])` for defining retarded time-delay systems.

.. note::
    :collapsible:

    While in the above syntax, the system matrices are provided as a list of 2D NumPy arrays, the function internally converts the system matrices into a 3D NumPy array, where the third dimension corresponds to the delays.
    The system matrices can also be created directly using the 3D NumPy array format, which is important when using the equivalent low-level API functions. 
    Equivalent to the above code snippet, the following code snippet also creates the system matrices can be defined using the 3D NumPy array format as:

    .. code-block:: python

        A = np.stack([A0, A1], axis=2)
        rdde = tds.RDDE(A=A, hA=delays)

    or alternately, the system matrices can be explicitly defined as:

    .. code-block:: python

        A = np.zeros(shape=(4,4,2))
        A[:,:,0] = A0
        A[:,:,1] = A1
        rdde = tds.RDDE(A=A, hA=delays)


For more details refer :ref:`tds_control_manual_examples`.


Neutral DDEs
-----------------------------

Similar to the retarded case, neutral systems with multiple point-wise delays are represented as delay-differential equations (DDEs) of the form:

.. math::
    :label: eq_ndde

    \dot{x}(t) + \sum_{i=1}^{m_H} H_i \dot{x}(t - h_{H_i}) = A_0 x(t) + \sum_{i=1}^{m_A} A_i x(t - h_{A_i})

where :math:`x(t) \in \mathbb{C}^n` is the state vector at time :math:`t`, :math:`A_0, A_1, \ldots, A_m` and :math:`H_1, H_2, \ldots, H_m` are constant matrices of size :math:`n \times n`, corresponding to delays :math:`h_{A_1}, \ldots, h_{A_m}` and :math:`h_{H_1}, \ldots, h_{H_m}` respectively.
The associated delay-difference equation (ADDE) for the above is given by:

.. math::
    :label: eq_adde

    x(t) + \sum_{i=1}^{m_H} H_i x(t - h_{H_i}) = 0.

In case of neutral systems, the above ADDE is important for the stability analysis of the system.

A neutral time-delay system can be created using the :func:`tdspy.ndde` class as demonstrated in the example below. 

.. admonition:: Example: See Example 1.18 from :cite:`michielsStability2007`
    :class: example

    Consider a neutral time-delay system, described by the DDE:

    .. math::

        \dot{x}(t) - \frac{3}{4} x(t-1) + \frac{1}{2} x(t-2) = \frac{1}{4} x(t) + \frac{3}{4} x(t - 2)

    We create the respective :class:`tdspy.ndde` object using the code:

    .. code-block:: python

        A0 = np.array([[0.25]])
        A1 = np.array([[0.75]])
        hA = np.array([0.,2.])
        H1 = np.array([[-0.75]])
        H2 = np.array([[0.5]])
        hH = np.array([1.,2.])
        ndde = tds.NDDE(H = [H1,H2], hH = hH, A = [A0, A1], hA=hA)

    To obtain the associated delay-difference equation, we can use the in-built method :func:`get_delay_difference_equation` from the :class:`tdspy.ndde` class as:

    .. code-block:: python

        adde = ndde.get_delay_difference_equation()

    Alternately, one can use the low-level API :func:`ndde_to_diff` function from the `tdspy.common.delay_difference_equation` module as:

    .. code-block:: python

        adde = ndde_to_diff(H=ndde.H, hH=ndde.hH)


Delay-differential algebraic equations (DDAEs)
-----------------------------------------------

A linear delay descriptor system can be modeled as delay-differential-algebraic equations (DDAEs) that take the form:

.. math::
    :label: eq_ddae

    E \dot{x}(t) = A_0 x(t) + \sum_{i=1}^{m} A_i x(t - h_i)

where :math:`x(t) \in \mathbb{C}^n` is the state vector, :math:`A_0, \ldots, A_m` are the respective system matrices, corresponding to delays :math:`h_1, \ldots, h_m` respectively. 
The above definition resembles the retarded case, with the difference that the matrix :math:`E` is allowed to be singular.

In the case of DDAEs, the system dynamics are described by a combination of differential and algebraic equations, which can lead to more complex behavior compared to retarded systems.
It can be shown also, that the above DDAE can be used to represent delay-differential equations of both retarded and neutral type and therefore, the DDAE framework 
is a more general framework used for representing a wide class of time-delay systems.  

In ``TDSpy``, a DDAE can be created using the :class:`tdspy.ddae` class.

.. admonition:: Example: Defining a DDAE
    :class: example

    Consider the following delay descriptor system with two delays:

    .. math::

        E \dot{x}(t) = A_0 x(t) + A_1 x(t - h_1)

    where :math:`E`, :math:`A_0` and :math:`A_1` are constant matrices, and :math:`h_1` is the delay.
    To create this system in ``TDSpy``, we use the following code snippet:

    .. code-block:: python

        E = np.array([[1, 0],
                        [0, 0]])
        A0 = np.array([[-1, 0],
                        [0, -2]])
        A1 = np.array([[0.5, 0],
                        [0, 0.5]])
        delays = np.array([0,1])
        ddae = tds.DDAE(E=E, A = [A0, A1], hA=delays)

    To identify the retarded or neutral nature, we can use the in-built method `is_essentially_retarded` (or alternately `is_essentially_neutral`) from the :class:`tdspy.ddae` class:

    .. code-block:: python

        is_retarded = ddae.is_essentially_retarded()

    Upon execution, it can be seen that the above system is neutral. Consequently, the system consists of a delay-difference equation which can 
    be extracted similar to the neutral case using the `get_delay_difference_equation` function from the :class:`tdspy.ddae` class as:

    .. code-block:: python

        adde = ddae.get_delay_difference_equation()

    Alternately, one can use the low-level API :func:`ddae_to_diff` function from the `tdspy.common.delay_difference_equation` module as:

    .. code-block:: python

        adde = ddae_to_diff(E = ddae.E, A=ddae.A, hA=ddae.hA)
