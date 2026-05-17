Installation
============

Requirements
------------

* Python 3.7 or higher
* ``ase`` — Atomic Simulation Environment
* ``numpy``, ``scipy``, ``matplotlib``, ``pandas``
* ``scikit-learn``
* ``dscribe``
* ``numba``
* ``pymatgen``

Install from PyPI
-----------------

The easiest way to install AtomicAI is via pip:

.. code-block:: bash

   pip install AtomicAI

Upgrade an existing installation:

.. code-block:: bash

   pip install --upgrade AtomicAI

Install from Source
-------------------

To install the latest development version directly from GitHub:

.. code-block:: bash

   git clone https://github.com/selvachandrasekaranselvaraj/AtomicAI.git
   cd AtomicAI
   pip install -e .

Verify Installation
-------------------

.. code-block:: bash

   python -c "import AtomicAI; print('AtomicAI installed successfully')"
   generate_descriptors --help
