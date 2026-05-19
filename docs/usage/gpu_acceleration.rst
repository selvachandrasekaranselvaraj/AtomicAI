GPU Acceleration
================

AtomicAI supports GPU-accelerated descriptor computation using CUDA via the
`numba.cuda <https://numba.readthedocs.io/en/stable/cuda/index.html>`_ JIT
compiler. When a CUDA-capable GPU is detected at import time, the
descriptor kernels run on the GPU automatically — no changes to your
existing scripts are required.

How it works
------------

The descriptor modules detect GPU availability once at startup:

.. code-block:: python

   # Internally, AtomicAI checks at import time:
   from numba import cuda
   _CUDA_AVAILABLE = cuda.is_available()

When ``_CUDA_AVAILABLE`` is ``True``, the ``create()`` method of each
descriptor class dispatches to a GPU kernel instead of the CPU path.
When ``False`` (no GPU, or CUDA toolkit not installed), the original
CPU ``numba @jit`` path is used transparently.

Accelerated components
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module / class
     - GPU-accelerated functions
   * - ``ACSF`` (``acsf.py``)
     - G2, G3, G4, G5 kernels — one GPU thread per center atom
   * - ``MBSF`` (``mbsf.py``)
     - GR (two-body) and GA (three-body) kernels
   * - ``calculate_fingerprint_vector`` (``laaf.py``)
     - Botu radial fingerprint — all atoms computed in a single kernel launch

The GPU parallelism strategy is the same for all kernels: each CUDA thread
is assigned one center atom and independently computes its full fingerprint
vector. Because every thread writes to its own output row, no atomic
operations or shared memory synchronisation are needed.

Requirements
------------

* CUDA-capable NVIDIA GPU (compute capability ≥ 3.5 recommended)
* CUDA Toolkit 11.x or 12.x
* ``numba >= 0.57`` compiled with CUDA support
* ``cuda-python >= 12.0`` (optional — provides low-level CUDA Python bindings)

.. note::

   The CPU path requires only ``numba`` (no GPU). The GPU path is an
   opt-in enhancement that activates automatically when the hardware and
   drivers are present.

Installation
------------

Using pip
~~~~~~~~~

.. code-block:: bash

   pip install "AtomicAI[cuda]"

This installs the ``cuda`` extras: ``numba>=0.57`` and ``cuda-python>=12.0``.

Using conda (recommended for CUDA toolkit management)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda install -c conda-forge numba cudatoolkit

Then install AtomicAI normally:

.. code-block:: bash

   pip install AtomicAI

Verifying GPU support
---------------------

Check whether AtomicAI sees your GPU:

.. code-block:: python

   from AtomicAI.descriptors.acsf import _CUDA_AVAILABLE
   print("CUDA available:", _CUDA_AVAILABLE)

Or from the command line:

.. code-block:: bash

   python -c "from AtomicAI.descriptors.acsf import _CUDA_AVAILABLE; print('CUDA:', _CUDA_AVAILABLE)"

You can also query numba directly:

.. code-block:: python

   from numba import cuda
   print(cuda.gpus)          # List detected GPUs
   print(cuda.is_available()) # True / False

Usage
-----

No code changes are needed. Run the same commands as usual — GPU
acceleration is applied automatically when available:

.. code-block:: bash

   # Runs on GPU if CUDA is available, CPU otherwise
   generate_descriptors trajectory.xyz --descriptor ACSF_G2G4 --n-eta 60

   # Force descriptor generation also benefits from GPU acceleration
   generate_force_descriptors trajectory.xyz --fp-type Split2b3b_ss

From Python:

.. code-block:: python

   from AtomicAI.descriptors.acsf import ACSF
   import numpy as np

   descriptor = ACSF(
       cutoff_descriptor=6.0,
       params_g2=params,
       species=['Li', 'O'],
   )
   # create() dispatches to GPU automatically if CUDA is available
   fingerprints = descriptor.create(atoms)

Performance notes
-----------------

* **G2 / G3 / GR (two-body)**: Scale as O(N) per center atom; benefit
  greatly from GPU parallelism for trajectories with hundreds or thousands
  of atoms.
* **G4 / G5 / GA (three-body)**: Scale as O(N²) per center atom; GPU
  parallelism provides the largest relative speedup for these
  computationally expensive terms.
* **Data transfer overhead**: For very small systems (< 50 atoms), CPU
  ``@jit`` may be faster because the GPU kernel launch and data transfer
  cost is not amortised. The GPU path is most beneficial for large
  simulation cells and long trajectories.

Troubleshooting
---------------

CUDA not detected
~~~~~~~~~~~~~~~~~

If ``_CUDA_AVAILABLE`` is ``False`` despite having an NVIDIA GPU:

1. Verify the driver is installed: ``nvidia-smi``
2. Verify the CUDA toolkit version matches numba's requirements:
   ``nvcc --version``
3. Check numba can see the GPU: ``python -c "from numba import cuda; print(cuda.gpus)"``
4. Re-install numba with CUDA support:
   ``conda install -c conda-forge numba cudatoolkit``

Out-of-memory errors
~~~~~~~~~~~~~~~~~~~~

For very large systems the device arrays may exceed GPU memory. In this
case the code will raise a ``numba.cuda.cudadrv.error.CudaDriverError``.
You can force CPU execution by temporarily setting:

.. code-block:: python

   import AtomicAI.descriptors.acsf as _acsf_mod
   _acsf_mod._CUDA_AVAILABLE = False
