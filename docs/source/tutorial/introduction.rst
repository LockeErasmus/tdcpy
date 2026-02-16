Introduction
============

.. contents:: Contents
   :local:
..    :depth: 2

.. As of now, I will put testing stuff here:


This tutorial will guide users through the important features of ``TDSpy`` with illustrations on how to use them. 
For detailed information and syntax on the individual functions and classes, please refer to the :doc:`../reference/api_reference` section of the
``TDSpy`` manual and in the references listed at the end of this document.      
The related theory on the topic, can be found in :cite:`michielsStability2007` and the ``TDS-Control`` manual :cite:`appeltans2023analysis`.


Getting Started
---------------

The most basic data structure utilized within ``TDSpy`` is the `NumPy <https://numpy.org/>`_ array. At the beginning of every script,
it is required to import the ``NumPy`` and ``TDSpy`` packages as follows:

.. code-block:: python

    import numpy as np
    import tdspy as tds


The secondary package `SciPy <https://scipy.org/>`_ is used for scientific computations and optimization, and can be imported as follows:

.. code-block:: python

    import scipy as sp

Similarly, the plotting package `Matplotlib <https://matplotlib.org/>`_ is used for visualization of results, and can be imported as follows:

.. code-block:: python

    import matplotlib.pyplot as plt

Once the packages have been imported, the user can create ``TDSpy`` objects, perform spectral analysis and design appropriate controllers. 
In the remaining part of this tutorial, we assume that the above main packages have been imported already.

Kindly follow the below links for the next steps of this guide.

==================================    ==============================================================
Section                                 Description
==================================    ==============================================================
:doc:`definitions`                      Defining a time-delay system
:doc:`analysis`                         Spectral analysis of time-delay systems 
:doc:`controller_design`                Stabilization and controller design
:doc:`../reference/api_reference`       Detailed description of all functions and classes
:ref:`general_examples`                 Examples of usage of TDSpy
==================================    ==============================================================

.. 1. To get acquainted with the basic functionality of ``TDSpy`` kindly follow the below tutorial sections:
    
..     * Creating a TDS: :ref:`tds_create`
..     * Spectral analysis of time-delay systems: :ref:`analysis`
..     * Spectral discretization: :ref:`discretization`
..     * Controller design for time-delay systems: :ref:`controller_design`

.. 2. For a detailed description of all functions and classes available in ``TDSpy``, refer: :ref:`api_reference`.

.. 3. Examples of usage can be found in :ref:`general_examples`.

High-level API vs Low-level API
--------------------------------

``TDSpy`` functions can be called using the low-level API or the high-level API. 
The high-level API is most often sufficient for users wanting to work directly with high-level functions, for example, root computation or controller design.
Internally, the high-level API wraps the low-level API around high-level functions to provide a simplified interface to users.
Nevertheless, for users intending to gain a deeper insight in the workings of the software, or for those wanting to include the basic functions 
into their own code, for example in optimization routines, users can directly access the low-level API, which provides the necessary building blocks.

For example, for computing the characteristic roots of a time-delay system, the below code snippet uses the **high-level** function :func:`tdspy.roots`, to provide a similar 
functionality as the TDS-Control MATLAB function `tds_roots`.

.. code-block:: python

    A0 = np.array([ [-1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, -10, -4],
                    [0, 0, 4, -10]])
    A1 = np.array([ [3, 3, 3, 3],
                    [0, -1.5, 0, 0],
                    [0, 0, 3, -5],
                    [0, 5, 5, 5]])
    hA = np.array([0,1])
    rdde = tds.RDDE(A = [A0, A1], hA=hA)
    roots, info = tds.roots(rdde, r=-1.0)

Computing the roots can alternately be performed using the **low-level** function :func:`roots_ddae` from the submodule `tdspy.stability.characteristic_roots` as follows:

.. code-block:: python

    import numpy as np
    from tdspy.stability.characteristic_roots import roots_ddae
    E = np.array(np.eye(4))
    A = np.zeros(shape=(4,4,2))
    A[:,:,0] = np.array([[-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -10, -4],
                        [0, 0, 4, -10]])
    A[:,:,1] = np.array([[3, 3, 3, 3],
                        [0, -1.5, 0, 0],
                        [0, 0, 3, -5],
                        [0, 5, 5, 5]])
    hA = np.array([0,1])
    cr, info = roots_ddae(E,A,hA,r=-1)

Both code snippets will yield the same results. However, the first snippet is practical if one is only interested in the characteristic roots of the time-delay system.
If however, the task of root computation is part of a larger framework, for example in an optimization routine,
the second code snippet provides more flexibility to the user in incorporating the root computation into their own code. 
Secondly, notice that the low-level API requires the user to explicitly provide the system matrices in a 3D NumPy array, whereas the high-level API handles this internally when creating the time-delay system object.    
More details on the structure of the ``TDSpy`` object can be found in the next sections of this tutorial.





.. minigallery:: 
    
    ..examples/tsds_control_manual/example_2_01.py



