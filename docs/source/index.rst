TDSpy Documentation
===========================


.. grid:: 1 1 2 2

   .. grid-item::
      :padding: 2
      :columns: 7

      **TDSpy: Control of time-delay systems in Python**

      **TDSpy** is a package for analysis and control of time-delay systems. 
      The software is based on the ``TDS-Control`` toolbox in MATLAB :cite:`appeltans2023analysis`.
      Like ``TDS-Control``, ``TDSpy`` provides tools for modeling, spectral analysis and controller design of time-delay systems.

      * :doc:`Modelling <tutorial/definitions>` time-delay systems 
      * Performing :doc:`spectral analysis <tutorial/analysis>`
      * Designing :doc:`stabilizing controllers <tutorial/controller_design>`
      * And :doc:`more <auto_examples/index>`!

   .. grid-item::
      :padding: 2
      :columns: 5

      .. image:: /_static/strong_stability.gif
         :align: center
         :width: 100%
         :target: ./auto_examples/stabopt/example_strong_sa.html
         

.. grid:: 1 1 2 2

   .. grid-item-card:: 
      :padding: 2
      :columns: 3
      :link: installation
      :link-type: doc
      :text-align: center
      :class-card: sd-v-stretch

      .. div:: sd-text-center 
         
         :octicon:`download;4em;sd-text-info`

      **Installation**

   .. grid-item-card::
      :padding: 2
      :columns: 3
      :link: tutorial/index
      :link-type: doc
      :text-align: center
      :class-card: sd-v-stretch

      .. div:: sd-text-center 
         
         :octicon:`book;4em;sd-text-info`
      
      **User-Guide**

   .. grid-item-card::
      :padding: 2
      :columns: 3
      :link: reference/api_reference   
      :link-type: doc
      :text-align: center
      :class-card: sd-v-stretch  

      .. div:: sd-text-center 
         
         :octicon:`code;4em;sd-text-info`

      **API Reference**

   .. grid-item-card::
      :padding: 2
      :columns: 3
      :link: auto_examples/index
      :link-type: doc
      :text-align: center
      :class-card: sd-v-stretch

      .. div:: sd-text-center 

         :octicon:`graph;4em;sd-text-info`
      
      **Examples**


.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   User Guide <tutorial/index>
   API Reference <reference/api_reference>
   Examples <auto_examples/index>
   credits
