
User Guide
==========

.. Welcome to the user guide for **tdcpy**!

``tdcpy`` provides an interface for modeling LTI time-delay systems, supporting systems of retarded type,
neutral type and delay descriptor systems. Next to modeling, ``tdcpy`` offers functionalities for spectral analysis, 
and controller design.

``tdcpy`` is based on the ``TDS-Control`` MATLAB toolbox, developed and published by Pieter Appeltans and Wim Michiels in KU Leuven :cite:`appeltans2023analysis`.
Users already familiar with ``TDS-Control`` will find the basic functionality similar.
``tdcpy`` additionally utilizes the Python libraries ``NumPy`` for arrays and array operations, ``Scipy`` for scientific computations and ``Matplotlib`` for plotting.
The user is required to have some basic familiarity with these packages when using ``tdcpy``.


Organization
---------------

``tdcpy`` is organized in several subpackages, each providing a specific set of functionalities. 
The main subpackages are:

===================  ==============================================
`tdcpy.common`       Core data structures and functions
`tdcpy.stability`    Functions for stability analysis
`tdcpy.plot`         Visualization tools
`tdcpy.stabopt`      Tools for stabilization and controller design
===================  ============================================== 

Besides these subpackages, several high-level modules are provided for direct usage:

=========================  ===============================================================================
`tdcpy.base`                  Base classes and functions
`tdcpy.closed_loop`           Functions for modeling the closed-loop.
`tdcpy.controller`            Functions for controller design.
`tdcpy.dae`                   Functions for modeling DAEs.
`tdcpy.ddae`                  Functions for modeling DDAEs.
`tdcpy.gamma`                 Functions for computing gamma_r.
`tdcpy.ndde`                  Functions for modeling neutral TDS.
`tdcpy.roots`                 Function for computing the characteristic roots.
`tdcpy.spectral_abscissa`     Functions for computing the spectral abscissa and strong spectral abscissa.
`tdcpy.zeros`                 Functions for computing the transmission zeros.
=========================  ===============================================================================

Kindly refer to the individual sections for more details on each of these subpackages and modules.


Quick-start
-----------

The following sections will guide you through the basic functionalities of ``tdcpy``.

.. toctree::
   :maxdepth: 2

   introduction
   definitions
   analysis
   controller_design
   references


