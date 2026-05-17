Quick Start
===========

This page shows the most common workflows in a few commands.

1. Convert a structure file
---------------------------

.. code-block:: bash

   vasp2xyz POSCAR          # VASP → XYZ
   vasp2cif POSCAR          # VASP → CIF
   xyz2vasp structure.xyz   # XYZ → VASP

2. Compute atomic descriptors
------------------------------

.. code-block:: bash

   # G2 radial descriptor (default)
   generate_descriptors trajectory.xyz

   # G2 + G4 angular descriptor, 60 eta functions
   generate_descriptors trajectory.xyz --descriptor ACSF_G2G4 --n-eta 60

   # Multiple descriptor types at once
   generate_descriptors trajectory.xyz --descriptor ACSF_G2 ACSF_G2G4G5 SOAP MBSF

3. Generate force descriptors
------------------------------

.. code-block:: bash

   # Default: Split2b3b_ss fingerprint
   generate_force_descriptors trajectory.xyz

   # Behler-Parrinello 2-body with custom cutoff
   generate_force_descriptors trajectory.xyz --fp-type BP2b --rc 8.0 --n2b 30

4. Radial Distribution Function
---------------------------------

.. code-block:: bash

   rdf trajectory.xyz
   rdf traj1.xyz traj2.xyz   # Compare two trajectories

5. Build a multilayer structure
---------------------------------

.. code-block:: bash

   build_multilayers POSCAR_A POSCAR_B
   build_multilayers POSCAR_A POSCAR_B POSCAR_C   # Trilayer

6. Generate surface slabs
--------------------------

.. code-block:: bash

   surfaces POSCAR   # Generates (100), (010), (001), (110), (101), (011), (111)

7. Dimensionality reduction
----------------------------

.. code-block:: bash

   pca                # Principal Component Analysis
   lpp                # Locality Preserving Projection
   dim_reduction      # Full pipeline (PCA + LPP + TsLPP)
