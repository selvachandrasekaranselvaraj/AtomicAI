File Format Conversion
======================

AtomicAI provides command-line tools to convert between common atomic structure
file formats. All tools read from ``sys.argv`` and write output to the current
directory.

VASP conversions
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``vasp2xyz POSCAR``
     - Convert VASP POSCAR/CONTCAR to XYZ
   * - ``vasp2cif POSCAR``
     - Convert VASP to CIF
   * - ``vasp2cq POSCAR``
     - Convert VASP to Conquest format
   * - ``vasp2lmp_data POSCAR``
     - Convert VASP to LAMMPS data file
   * - ``vasp2vasp POSCAR``
     - Re-write VASP file (useful for sorting/reformatting)
   * - ``xyz2vasp structure.xyz``
     - Convert XYZ to VASP POSCAR
   * - ``lmp2vasp dump.lammps``
     - Convert LAMMPS trajectory dump to VASP

Conquest conversions
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Description
   * - ``cq2vasp file.cq``
     - Convert Conquest to VASP
   * - ``cq2cif file.cq``
     - Convert Conquest to CIF
   * - ``cif2cq file.cif``
     - Convert CIF to Conquest

Trajectory conversions
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Description
   * - ``ase_traj2xyz_traj traj.traj``
     - Convert ASE binary trajectory to XYZ trajectory
   * - ``wrap2unwrap POSCAR``
     - Convert wrapped (periodic) coordinates to unwrapped

Examples
--------

.. code-block:: bash

   # Convert a POSCAR to XYZ for visualization
   vasp2xyz POSCAR

   # Convert a CIF from a database to VASP for DFT input
   # (not directly available — use cif2cq then cq2vasp, or ase directly)

   # Convert LAMMPS dump to VASP for post-processing
   lmp2vasp dump.lammps
