Structure Analysis
==================

Radial Distribution Function
-----------------------------

Compute and plot the RDF from one or more trajectory files:

.. code-block:: bash

   rdf trajectory.xyz
   rdf traj1.xyz traj2.xyz traj3.xyz   # Overlay multiple trajectories

Supported formats: ``.xyz``, LAMMPS dump (``.lmp``).

The tool automatically:

* Detects all element pairs present in the trajectory
* Selects ``predicted`` pairs (cation-anion) for complex multi-component systems,
  or all ``available`` pairs for simple systems
* Applies a moving-average smoothing (window = 10 bins)
* Saves the figure as ``rdf.png``

Structure analysis
------------------

General structural analysis (coordination numbers, bond lengths, angles):

.. code-block:: bash

   structure_analysis trajectory.xyz

Molecular dynamics statistics
------------------------------

Plot energy, temperature, pressure, and volume from MD runs:

.. code-block:: bash

   plot_md_stats          # Generic MD stats
   plot_vasp_md           # VASP OUTCAR / vasprun.xml
   plot_lammps_md         # LAMMPS log file
