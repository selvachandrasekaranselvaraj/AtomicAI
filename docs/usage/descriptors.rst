Atomic Descriptors
==================

AtomicAI provides locally-averaged atomic fingerprints (LAAF) that encode the
chemical environment of each atom. These descriptors are used as input features
for machine learning models.

Command-line usage
------------------

.. code-block:: bash

   generate_descriptors trajectory.xyz [--descriptor TYPE [TYPE ...]] [--n-eta N]

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Option
     - Default
     - Description
   * - ``--descriptor`` / ``-d``
     - ``ACSF_G2 ACSF_G2G4 SOAP``
     - One or more descriptor types to compute
   * - ``--n-eta`` / ``-n``
     - ``50``
     - Number of eta decay functions

Descriptor types
----------------

ACSF_G2 — Radial symmetry functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two-body Behler-Parrinello G2 functions. Each function is a Gaussian in
interatomic distance, parameterised by eta (width) and R\ :sub:`s` (shift):

.. math::

   G^2_i = \sum_j e^{-\eta (r_{ij} - R_s)^2} \cdot f_c(r_{ij})

where :math:`f_c` is a cosine cutoff function.

.. code-block:: bash

   generate_descriptors traj.xyz --descriptor ACSF_G2 --n-eta 80

ACSF_G3 — Cosine basis functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

G3 functions use a cosine basis parameterised by kappa:

.. math::

   G^3_i = \sum_j \cos(\kappa \cdot r_{ij}) \cdot f_c(r_{ij})

.. code-block:: bash

   generate_descriptors traj.xyz --descriptor ACSF_G3

ACSF_G4 — Angular symmetry functions (with r\ :sub:`jk`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three-body functions that encode bond angles. The r\ :sub:`jk` term is included
in the cutoff and exponent sum:

.. math::

   G^4_i = 2^{1-\zeta} \sum_{j,k \neq i} (1 + \lambda \cos\theta_{jik})^\zeta
            \cdot e^{-\eta(r_{ij}^2 + r_{ik}^2 + r_{jk}^2)} \cdot f_c(r_{ij}) f_c(r_{ik}) f_c(r_{jk})

.. code-block:: bash

   generate_descriptors traj.xyz --descriptor ACSF_G4

ACSF_G5 — Angular symmetry functions (without r\ :sub:`jk`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to G4 but the r\ :sub:`jk` distance is not included, making it faster
to compute for large systems:

.. math::

   G^5_i = 2^{1-\zeta} \sum_{j,k \neq i} (1 + \lambda \cos\theta_{jik})^\zeta
            \cdot e^{-\eta(r_{ij}^2 + r_{ik}^2)} \cdot f_c(r_{ij}) f_c(r_{ik})

.. code-block:: bash

   generate_descriptors traj.xyz --descriptor ACSF_G5

ACSF_G2G4 — Combined radial + angular (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Concatenates G2 and G4 vectors to produce a complete two-body + three-body
descriptor. This is generally the best balance of accuracy and cost.

.. code-block:: bash

   generate_descriptors traj.xyz --descriptor ACSF_G2G4 --n-eta 60

ACSF_G2G4G5 — Full combined descriptor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Concatenates G2 + G4 + G5. Provides the richest angular description.

.. code-block:: bash

   generate_descriptors traj.xyz --descriptor ACSF_G2G4G5

SOAP — Smooth Overlap of Atomic Positions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rotationally invariant descriptor based on the overlap of atomic density
functions, computed via `DScribe <https://singroup.github.io/dscribe/>`_.

.. code-block:: bash

   generate_descriptors traj.xyz --descriptor SOAP

MBSF — Many-body symmetry functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combines a radial term (gr, G2-like) with an angular term (ga) that includes
:math:`\zeta`, :math:`\theta_s`, eta, and R\ :sub:`s` parameters.

.. code-block:: bash

   generate_descriptors traj.xyz --descriptor MBSF

Output
------

Descriptor files are written to ``./descriptors/`` with the naming convention::

   <TYPE>_<cutoff_descriptor>_<cutoff_average>_<element1>_<element2>.dat

Each row is one averaged fingerprint vector for a single atom. The cutoff values
(in Å) come from the built-in ``descriptor_cutoff`` table in
``AtomicAI/data/data_lib.py``.

Running multiple types
----------------------

You can compute several descriptor types in a single call — they run in parallel
using Python multiprocessing:

.. code-block:: bash

   generate_descriptors traj.xyz --descriptor ACSF_G2 ACSF_G2G4 SOAP MBSF --n-eta 50
