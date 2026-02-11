Spectral analysis
==================


Unlike ordinary differential equations, the spectrum of time-delay systems consists of infinitely many characteristic roots.
To this end, `TDSpy` provides tools for computing and plotting the characteristic roots.

We distinguish between systems of retarded and neutral type.


Spectrum of a retarded system
------------------------------

The characteristic equation for a retarded time delay system can be written as

.. math::

    \det\left(\lambda I - A_0 - \sum_{i=1}^m A_i e^{-\lambda \tau_i}\right) = 0.

Unlike ODEs, the characteristic function is a quasipolynomial, which in general consists of infinitely 
many characteristic roots. However it is known that for retarded systems, there can exist only finitely many 
roots in a given half plane. 

`TDSpy` follows the procedure described in :cite:`wu2012reliably` for computing the characteristic roots of retarded systems.
The procedure consists of two steps: first, a finite number of characteristic roots are computed using a spectral discretization method, following which the roots are refined using a Newton-type iteration.
The tutorial :ref:`_tutorials` provides a step-by-step guide into the discretization procedure.

In `TDSpy`, the characteristic roots of a retarded system can be computed using the :func:`tdspy.roots` function. Alternately, one can also use the low-level API function :func:`tdspy.roots_ddae` for computing the characteristic roots in a given region of the complex plane. 

