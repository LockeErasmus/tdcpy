Introduction
============

.. contents:: Contents
   :local:
..    :depth: 2

.. As of now, I will put testing stuff here:
``TDSpy`` is a Python package for analysis and control of time-delay systems. 
The package provides tools for modeling of time-delay systems, supporting systems of retarded type,
neutral type and delay descriptor systems. Next to modeling, ``TDSpy`` offers functionalities for spectral analysis, 
and controller design.

The software is modelled around the ``TDS-Control`` MATLAB toolbox developed and published by the team of Wim Michiels in KU Leuven.
Users already familiar with ``TDS-Control`` will find the basic functionality similar to the MATLAB toolbox.
``TDSpy`` however utilizes additional Python libraries, mainly ``NumPy`` for arrays and array operations, ``Scipy`` for scientific computations and ``Matplotlib`` for plotting.

This tutorial will guide users through the important features of ``TDSpy`` and will illustrate how to use them.
For more detailed information on the individual functions and classes, please refer to the :doc:`../reference/api_reference` section of the
``TDSpy`` manual and in the references listed at the end of this document.      
For the related theory on the topic, we refer the ``TDS-Control`` manual :cite:`appeltans2023analysis`.


Getting Started
---------------

The most basic data structure utilized within ``TDSpy`` is the `NumPy <https://numpy.org/>`_ array. At the beginning of every script,
the user is required to import the ``NumPy`` and ``TDSpy`` packages as follows:

    >>> import numpy as np
    >>> import tdspy as tds


The secondary package `SciPy <https://scipy.org/>`_ is used for advanced scientific computations, and can be imported as follows:

    >>> import scipy as sp

Similarly, the plotting package `Matplotlib <https://matplotlib.org/>`_ is used for visualization of results, and can be imported as follows:

    >>> import matplotlib.pyplot as plt

Once the packages have been imported, the user can then proceed to create ``TDSpy`` objects, perform spectral analysis and controller design. 
Kindly refer to the below sections for an overview of the contents.

==================================    ==============================================================
Section                                 Description
==================================    ==============================================================
:doc:`definitions`                      Defining a time-delay system
:doc:`analysis`                         Spectral analysis of time-delay systems 
:doc:`discretization`                   Spectral discretization
:doc:`controller_design`                Stabilization and controller design
:doc:`../reference/api_reference`       Detailed description of all functions and classes available
:ref:`general_examples`                 Examples of usage of TDSpy
==================================    ==============================================================

.. 1. To get acquainted with the basic functionality of ``TDSpy`` kindly follow the below tutorial sections:
    
..     * Creating a TDS: :ref:`tds_create`
..     * Spectral analysis of time-delay systems: :ref:`analysis`
..     * Spectral discretization: :ref:`discretization`
..     * Controller design for time-delay systems: :ref:`controller_design`

.. 2. For a detailed description of all functions and classes available in ``TDSpy``, refer: :ref:`api_reference`.

.. 3. Examples of usage can be found in :ref:`general_examples`.

TDSpy Structure
--------------------

``TDSpy`` functions can be called using the low-level API or the high-level API. 
The high-level API is most often sufficient for most users wanting to work directly with high-level functionalities such as spectral analysis and stabilization.
Internally, the high-level API makes use of the low-level API to perform the necessary computations.
For users wanting to gain a deeper insight in the workings of the software, or wanting to include the basic functions 
into their own code, for example in optimization routines, users can directly access the low-level API, which provides the necessary building blocks.

.. High-level API
.. ~~~~~~~~~~~~~~~

.. * `tdspy.roots`: Function for computing the characteristic roots.
.. * `tdspy.spectral_abscissa`: Functions for computing the spectral abscissa and strong spectral abscissa.
.. * `tdspy.zeros`: Functions for computing the transmission zeros.
.. * `tdspy.closed_loop`: Functions for modeling the closed-loop.
.. * `tdspy.controller`: Functions for controller design.
.. * `tdspy.dae`: Functions for modeling delay-algebraic equations (DAE).
.. * `tdspy.ddae`: Functions for modeling delay-differential-algebraic equations (DDAE).
.. * `tdspy.gamma`: Functions for computing gamma_r.
.. * `tdspy.ndde`: Functions for modeling neutral time-delay systems.

.. Low-level API
.. ~~~~~~~~~~~~~

.. * `tdspy.common`: Core data structures and functions.
.. * `tdspy.stability`: Functions for stability analysis.
.. * `tdspy.stabopt`: Tools for stabilization and controller design.
.. * `tdspy.plot`: Visualization tools.

For example, for computing the characteristic roots of a time-delay system, the below code snippet uses the **high-level** function ``tds.roots``, to provide a similar 
functionality as the TDS-Control MATLAB function ``tds_roots``.

    >>> import numpy as np
    >>> import tdspy as tds
    >>> A0 = np.array([ [-1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, -10, -4],
                        [0, 0, 4, -10]])
    >>> A1 = np.array([ [3, 3, 3, 3],
                        [0, -1.5, 0, 0],
                        [0, 0, 3, -5],
                        [0, 5, 5, 5]])
    >>> hA = np.array([0,1])
    >>> rdde = tds.RDDE(A = [A0, A1], hA=hA)
    >>> roots, info = tds.roots(rdde, r=-1.0)

Computing the roots can alternately be performed using the **low-level** function ``roots_ddae`` from the submodule ``tdspy.stability.characteristic_roots`` as follows:

    >>> import numpy as np
    >>> from tdspy.stability.characteristic_roots import roots_ddae
    >>> E = np.array(np.eye(4))
    >>> A = np.zeros(shape=(4,4,2))
    >>> A[:,:,0] = np.array([[-1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, -10, -4],
                            [0, 0, 4, -10]])
    >>> A[:,:,1] = np.array([[3, 3, 3, 3],
                            [0, -1.5, 0, 0],
                            [0, 0, 3, -5],
                            [0, 5, 5, 5]])
    >>> hA = np.array([0,1])
    >>> cr, info = roots_ddae(E,A,hA,r=-1)

Both code snippets will yield the same results. However, the first snippet is practical if one is only interested in the characteristic roots of the time-delay system.
If however, the task of root computation is part of a larger framework, for example in an optimization routine,
the second code snippet provides more flexibility to the user in incorporating the root computation into their own code. 
Secondly, notice that the low-level API requires the user to explicitly provide the system matrices in a 3D NumPy array, whereas the high-level API handles this internally when creating the time-delay system object.    
More details on the structure of the ``TDSpy`` object can be found in the next sections of this tutorial.





.. minigallery:: 
    
    ..examples/tsds_control_manual/example_2_01.py



