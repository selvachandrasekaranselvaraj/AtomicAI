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

GPU Acceleration (optional)
---------------------------

To enable CUDA-accelerated descriptor computation, install AtomicAI with the
``cuda`` extras:

.. code-block:: bash

   pip install "AtomicAI[cuda]"

This pulls in ``numba>=0.57`` and ``cuda-python>=12.0``. A CUDA-capable NVIDIA
GPU and the CUDA Toolkit (11.x or 12.x) must be present on the system.

Alternatively, manage the CUDA toolkit through conda:

.. code-block:: bash

   conda install -c conda-forge numba cudatoolkit
   pip install AtomicAI

GPU support activates automatically at runtime — no code changes are needed.
See :doc:`usage/gpu_acceleration` for full details.

Verify Installation
-------------------

.. code-block:: bash

   python -c "import AtomicAI; print('AtomicAI installed successfully')"
   generate_descriptors --help

Check GPU support:

.. code-block:: bash

   python -c "from AtomicAI.descriptors.acsf import _CUDA_AVAILABLE; print('CUDA:', _CUDA_AVAILABLE)"
