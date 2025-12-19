Installation
============

This section describes how to install ``tdspy`` using ``pip`` or from source.
It is recommended to install the package inside a virtual environment.

Prerequisites
--------------

* Python 3.10 or newer
* ``pip`` installed

Creating a virtual environment
-------------------------------

First, create and activate a virtual environment.

On Linux and macOS:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate

On Windows:

.. code-block:: bash

   python -m venv venv
   venv\Scripts\activate

Once activated, your shell prompt should indicate that the virtual environment
is active.

Installing with pip
-------------------

The easiest way to install ``tdspy`` is via ``pip``:

.. code-block:: bash

   pip install tdspy

To upgrade to the latest version:

.. code-block:: bash

   pip install --upgrade tdspy


Installing from source
----------------------

To install ``tdspy`` from source, clone the GitHub repository:

.. code-block:: bash

   git clone https://github.com/LockeErasmus/tdspy.git
   cd tdspy

Install the package using ``pip``:

.. code-block:: bash

   pip install .

For development purposes, install the package in editable mode:

.. code-block:: bash

   pip install -e .
