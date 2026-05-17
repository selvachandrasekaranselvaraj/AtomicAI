Machine Learning Force Fields
==============================

AtomicAI supports the generation of machine learning force fields (MLFF) using
a linear regression approach (LassoLarsCV) applied to force-projected atomic
fingerprints.

Workflow
--------

The full MLFF workflow is:

1. **Prepare a trajectory** with forces (from VASP AIMD or LAMMPS)
2. **Generate force descriptors** — encode atomic environments
3. **Train the force field** — fit LassoLarsCV with variance thresholding and
   standard scaling
4. **Predict forces** — use the trained model via the ASE calculator interface

Step 1: Generate force descriptors
------------------------------------

.. code-block:: bash

   generate_force_descriptors trajectory.xyz --fp-type Split2b3b_ss --rc 10.5

This writes ``./descriptors/force_descriptors.dat``.

Step 2: Train the MLFF
-----------------------

The ``get_mlff`` function in ``AtomicAI.mlff.mlff`` reads the descriptor file,
splits data into training/test sets (80/20), applies variance thresholding,
standard scaling, and fits a LassoLarsCV model.

It evaluates multiple variance thresholds and selects the one that maximises R²
on the test set.

Step 3: LAMMPS input files
---------------------------

AtomicAI can generate LAMMPS input files for NPT and NVT molecular dynamics:

.. code-block:: bash

   lammps_npt_inputs    # Generate NPT input
   lammps_nvt_inputs    # Generate NVT input

VASP database inputs
---------------------

For generating systematic VASP input sets for database calculations:

.. code-block:: bash

   vaspDB_vc_run      # Variable-cell relaxation inputs
   vaspDB_aimd_run    # Ab initio MD inputs
