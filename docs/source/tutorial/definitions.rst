Definitions
==============================


Retarded DDEs
---------------------

A time-delay system of retarded type can be described by a set of delay-differential equations (DDEs) where the highest derivative of the state variable does not depend on delayed terms.
An LTI retarded (autonomous) TDS can be represented as a DDE the following general form:

.. math::

    \dot{x}(t) = A_0 x(t) + \sum_{i=1}^{m} A_i x(t - h_i)

where :math:`x(t) \in \mathbb{R}^n` is the state vector, :math:`A_0, A_1, \ldots, A_m` are constant matrices of size :math:`n \times n`, corresponding to delays :math:`h_1, h_2, \ldots, h_m` respectively, with :math:`0 < h_1 < h_2 < \ldots < h_m`.    

In ``TDSpy``, a retarded time-delay system can be created using the `tdspy.RDDE` class.
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

