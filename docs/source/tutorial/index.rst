
User Guide
==========

.. Welcome to the user guide for **tdspy**!
``TDSpy`` is a Python package for analysis and control of time-delay systems. 
The package provides tools for modeling of time-delay systems, supporting systems of retarded type,
neutral type and delay descriptor systems. Next to modeling, ``TDSpy`` offers functionalities for spectral analysis, 
and controller design.


Organization
---------------

``TDSpy`` is organized in several subpackages, each providing a specific set of functionalities. 
The main subpackages are:

===================  ==============================================
`tdspy.common`       Core data structures and functions
`tdspy.stability`    Functions for stability analysis
`tdspy.plot`         Visualization tools
`tdspy.stabopt`      Tools for stabilization and controller design
===================  ============================================== 

Besides these subpackages, several high-level modules are provided for direct usage:

=========================  ===============================================================================
`tdspy.base`                  Base classes and functions
`tdspy.closed_loop`           Functions for modeling the closed-loop.
`tdspy.controller`            Functions for controller design.
`tdspy.dae`                   Functions for modeling DAEs.
`tdspy.ddae`                  Functions for modeling DDAEs.
`tdspy.gamma`                 Functions for computing gamma_r.
`tdspy.ndde`                  Functions for modeling neutral TDS.
`tdspy.roots`                 Function for computing the characteristic roots.
`tdspy.spectral_abscissa`     Functions for computing the spectral abscissa and strong spectral abscissa.
`tdspy.zeros`                 Functions for computing the transmission zeros.
=========================  ===============================================================================

Kindly refer to the individual sections for more details on each of these subpackages and modules.


Quick-start Guide
-----------------

The following sections will guide you through the basic functionalities of ``TDSpy``.

.. toctree::
   :maxdepth: 1

   introduction
   definitions
   analysis
   controller_design
   references


