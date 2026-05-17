Force Descriptors
=================

Force descriptors encode the atomic environment in a form suitable for learning
the force component of a machine learning force field (MLFF). Each descriptor
is projected onto a random unit vector to give a scalar training target, following
the force-projection approach of Botu *et al.*

Command-line usage
------------------

.. code-block:: bash

   generate_force_descriptors trajectory.xyz [OPTIONS]

The trajectory must contain force data (``forces`` or ``momenta`` arrays).

Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Option
     - Default
     - Description
   * - ``--fp-type`` / ``-f``
     - ``Split2b3b_ss``
     - Fingerprint type (see below)
   * - ``--rc`` / ``-r``
     - ``10.5``
     - Cutoff radius in Å for both 2-body and 3-body terms
   * - ``--n2b``
     - ``20``
     - Number of 2-body eta functions
   * - ``--n3b``
     - ``10``
     - Number of 3-body eta functions

Fingerprint types
-----------------

BP2b — Behler-Parrinello 2-body
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Force-projected two-body fingerprint. Efficient for simple systems.

.. code-block:: bash

   generate_force_descriptors traj.xyz --fp-type BP2b --rc 8.0 --n2b 30

Split2b3b_ss — Split 2-body + 3-body (same species)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Combines a 2-body and a 3-body fingerprint restricted to same-species
neighbour pairs. Recommended for multi-component systems.

.. code-block:: bash

   generate_force_descriptors traj.xyz --fp-type Split2b3b_ss --rc 10.5 --n2b 20 --n3b 10

Output
------

Results are written to ``./descriptors/force_descriptors.dat``. Each row
contains the fingerprint vector followed by the force projection value and
the atomic symbol::

   <fp_1>  <fp_2>  ...  <fp_N>  <Fv>  <symbol>

This file is used directly as input to the MLFF training step.
