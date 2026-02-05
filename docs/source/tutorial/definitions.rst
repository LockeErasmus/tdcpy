Definitions
==============================

Time-Delay Systems
------------------------------

A time-delay system (TDS) is a dynamical system in which the evolution of the state depends not only on its current state but also on its past states.
Time-delay systems are commonly modeled using delay-differential equations (DDEs), which are differential equations that include terms with delays.
Time-delay systems can be classified into different types based on the nature of the delays and their influence on the system dynamics.

Time-delay systems can be broadly classified as follows:
- Retarded time-delay systems
- Neutral time-delay systems
- Delay descriptor systems



Retarded DDEs
---------------------

A (linear) time-delay system of retarded type can be described by a set of delay-differential equations (DDEs) where the highest derivative of the state variable does not depend on delayed terms.
An LTI retarded (autonomous) TDS can be represented as a DDE (RDDE) of the following general form:

.. math::

    \dot{x}(t) = A_0 x(t) + \sum_{i=1}^{m} A_i x(t - h_i)

where :math:`x(t) \in \mathbb{R}^n` is the state vector, :math:`A_0, A_1, \ldots, A_m` are constant matrices of size :math:`n \times n`, corresponding to delays :math:`h_1, h_2, \ldots, h_m` respectively, with :math:`0 < h_1 < h_2 < \ldots < h_m`.    

In ``TDSpy``, a retarded time-delay system can be created using the :func:`tdspy.RDDE` class.
For example, consider the following retarded time-delay system with two delays:

.. math::

    \dot{x}(t) = A_0 x(t) + A_1 x(t - h_1)

where :math:`A_0` and :math:`A_1` are constant matrices, and :math:`h_1` is the delay.
To create this system in ``TDSpy``, you can use the following code snippet:
    
    >>> import numpy as np
    >>> import tdspy as tds
    >>> A0 = np.array([[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -10, -4],
                        [0, 0, 4, -10]])
    >>> A1 = np.array([[3, 3, 3, 3],
                        [0, -1.5, 0, 0],
                        [0, 0, 3, -5],
                        [0, 5, 5, 5]])
    >>> delays = np.array([0,0.5])
    >>> rdde = tds.RDDE(A = [A0, A1], hA=delays)


Neutral DDEs
-----------------------------
A (linear) time-delay system of neutral type can be described by a set of delay-differential equations (DDEs) where the highest derivative of the state variable depends on delayed terms.
An LTI neutral (autonomous) TDS can be represented as a DDE (NDDE) of the following general form:

.. math::

    \frac{d}{dt}\left( x(t) + \sum_{i=1}^{m} H_i x(t - h_i) \right) = A_0 x(t) + \sum_{i=1}^{m} A_i x(t - h_i)

where :math:`x(t) \in \mathbb{R}^n` is the state vector, :math:`A_0, A_1, \ldots, A_m` and :math:`B_1, B_2, \ldots, B_m` are constant matrices of size :math:`n \times n`, corresponding to delays :math:`h_1, h_2, \ldots, h_m` respectively, with :math:`0 < h_1 < h_2 < \ldots < h_m`.

In ``TDSpy``, a neutral time-delay system can be created using the :func:`tdspy.NDDE` class.
For example, consider the following neutral time-delay system with two delays:
.. math::

    \frac{d}{dt}\left( x(t) + H_1 x(t - h_1) \right) = A_0 x(t) + A_1 x(t - h_1)

where :math:`A_0`, :math:`A_1` and :math:`H_1` are constant matrices, and :math:`h_1` is the delay.
To create this system in ``TDSpy``, you can use the following code snippet:

    >>> import numpy as np
    >>> import tdspy as tds
    >>> A0 = np.array([[-1, 0],
                        [0, -2]])
    >>> A1 = np.array([[0.5, 0],
                        [0, 0.5]])
    >>> H1 = np.array([[0.2, 0],
                        [0, 0.2]])
    >>> delays = np.array([0,1])
    >>> ndde = tds.NDDE(A = [A0, A1], H = [H1], hA=delays)


Delay Descriptor Systems
------------------------------
A delay descriptor system (DDS) is a type of time-delay system that combines features of both retarded and neutral time-delay systems.
A linear time-delay descriptor system can be by a set of delay-differential-algebraic equations (DDAEs) which take the general form:

.. math::

    E \dot{x}(t) = A_0 x(t) + \sum_{i=1}^{m} A_i x(t - h_i)

where :math:`E` is a singular matrix, :math:`x(t) \in \mathbb{R}^n` is the state vector, :math:`A_0, A_1, \ldots, A_m` are constant matrices of size :math:`n \times n`, corresponding to delays :math:`h_1, h_2, \ldots, h_m` respectively, with :math:`0 < h_1 < h_2 < \ldots < h_m`. 

In ``TDSpy``, a delay descriptor system can be created using the :func:`tdspy.DDAE` class.
For example, consider the following delay descriptor system with two delays:
.. math::

    E \dot{x}(t) = A_0 x(t) + A_1 x(t - h_1)

where :math:`E`, :math:`A_0` and :math:`A_1` are constant matrices, and :math:`h_1` is the delay.
To create this system in ``TDSpy``, you can use the following code snippet:
    >>> import numpy as np
    >>> import tdspy as tds
    >>> E = np.array([[1, 0],
                        [0, 0]])
    >>> A0 = np.array([[-1, 0],
                        [0, -2]])
    >>> A1 = np.array([[0.5, 0],
                        [0, 0.5]])
    >>> delays = np.array([0,1])
    >>> ddae = tds.DDAE(E=E, A = [A0, A1], hA=delays)
