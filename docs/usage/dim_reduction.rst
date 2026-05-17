Dimensionality Reduction
=========================

AtomicAI provides several dimensionality reduction methods to project
high-dimensional descriptor vectors into 2D or 3D spaces for visualisation
and clustering.

Available methods
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Method
   * - ``pca``
     - Principal Component Analysis
   * - ``lpp``
     - Locality Preserving Projection
   * - ``dim_reduction``
     - Full pipeline (PCA → LPP → TsLPP)
   * - ``dim_reduction_mpi``
     - Parallel version using MPI
   * - ``optimize_tslpp_hyperparameters_without_prediction``
     - TsLPP hyperparameter search (training only)
   * - ``optimize_tslpp_hyperparameters_with_prediction``
     - TsLPP hyperparameter search with test-set prediction
   * - ``predict_tslpp``
     - Apply a trained TsLPP model to new data

Usage
-----

.. code-block:: bash

   # PCA only
   pca

   # LPP only
   lpp

   # Full pipeline
   dim_reduction

   # Parallel pipeline (requires mpi4py)
   mpirun -n 8 dim_reduction_mpi

   # Optimise TsLPP hyperparameters then predict
   optimize_tslpp_hyperparameters_with_prediction
   predict_tslpp

TsLPP
-----

Temperature-scaled Locality Preserving Projection (TsLPP) is a supervised
variant of LPP that uses temperature labels to improve the separation of
structural phases in the projected space. It is particularly effective for
classifying amorphous, liquid, and crystalline phases.
