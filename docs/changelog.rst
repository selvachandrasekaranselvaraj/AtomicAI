Changelog
=========

0.5.0 (2026)
------------

**New features**

* **GPU acceleration** — CUDA-accelerated descriptor kernels via ``numba.cuda``
  for all ACSF (G2, G3, G4, G5), MBSF (GR, GA), and Botu (LAAF) fingerprint
  functions. GPU dispatch is automatic: if a CUDA-capable GPU is detected at
  import time, ``ACSF.create()`` and ``MBSF.create()`` use the GPU path;
  otherwise the existing CPU ``@jit`` path is used unchanged.
* ``setup.py``: added ``extras_require={'cuda': ['numba>=0.57', 'cuda-python>=12.0']}``
  so users can install GPU dependencies with ``pip install "AtomicAI[cuda]"``.
* New documentation page: *GPU Acceleration* (``usage/gpu_acceleration``).

**Improvements**

* ``ACSF.create()`` and ``MBSF.create()`` refactored into ``_create_cpu()``
  and ``_create_cuda()`` helpers for clarity and testability; public interface
  is unchanged.

0.4.0 (2025)
------------

**New features**

* ``generate_descriptors``: added ``--descriptor`` and ``--n-eta`` CLI options;
  new descriptor types ``ACSF_G3``, ``ACSF_G5``, ``ACSF_G2G4G5`` (in addition to
  existing ``ACSF_G2``, ``ACSF_G4``, ``ACSF_G2G4``, ``SOAP``, ``MBSF``)
* ``generate_force_descriptors``: added ``--fp-type``, ``--rc``, ``--n2b``,
  ``--n3b`` CLI options
* ``surfaces``: new command replacing ``create_surfaces``; generates surface slabs
  for (100), (010), (001), (110), (101), (011), (111) Miller planes

**Bug fixes**

* Fixed unresolved git merge conflicts in ``calculate_descriptors.py`` and
  ``select_snapshots.py`` (files were unimportable)
* Fixed ``NameError`` in ``surfaces.py`` (``miller_indices`` was undefined)
* Fixed incorrect supercell repeat count in ``rdf.py`` (while-loop accumulated
  products instead of scaling from original cell length)

**Improvements**

* Removed heavy pandas dependency from ``rdf.supercell()``; sorting now uses
  numpy ``argsort``
* Cleaned up ``build_multilayers.py``: merged duplicate module docstring,
  removed unused variable

0.3.0
-----

* Initial public release with VASP/CIF/XYZ/Conquest format conversion,
  RDF, structure analysis, ACSF G2/G2G4/SOAP descriptors, MLFF, and
  dimensionality reduction tools.
