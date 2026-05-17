Structure Building Tools
========================

supercell
---------

Create a supercell from a VASP structure:

.. code-block:: bash

   supercell POSCAR

build_multilayers
-----------------

Stack two or more VASP structures into a commensurate multilayer system.
The tool automatically finds the optimal supercell multiples that minimise
lattice mismatch (default tolerance: 4 %), then explores all non-redundant
stacking permutations.

.. code-block:: bash

   build_multilayers POSCAR_A POSCAR_B
   build_multilayers POSCAR_A POSCAR_B POSCAR_C   # Trilayer

For each permutation, a ``<name_A>_<name_B>.vasp`` file is written along
with a log file. The default interlayer gap is 2.3 Å.

build_constrained_multilayers
------------------------------

Similar to ``build_multilayers`` but allows explicit lattice constraints:

.. code-block:: bash

   build_constrained_multilayers POSCAR_A POSCAR_B

surfaces
--------

Generate surface slabs for common Miller indices from a bulk structure.
Slabs are created using pymatgen's ``SlabGenerator`` with a minimum vacuum
of 15 Å and minimum slab thickness of 8.5 Å.

.. code-block:: bash

   surfaces POSCAR

Default Miller indices generated: (100), (010), (001), (110), (101), (011), (111).
Output files are written to the ``surfaces/`` directory with naming::

   surfaces/<name>_<hkl>_<termination>.vasp

build_interface
---------------

Build an interface between two materials:

.. code-block:: bash

   build_interface POSCAR_A POSCAR_B
