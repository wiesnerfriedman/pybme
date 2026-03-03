"""PyBME Tutorials
==================

Python translations of the MATLAB BMElib ``tutorlib`` tutorials.
Each tutorial is a standalone runnable script.

Tutorials
---------

.. list-table::
   :header-rows: 1

   * - Tutorial
     - MATLAB equivalent
     - Topic
   * - ``tutorial_models``
     - MODELSLIBtutorial
     - Covariance & variogram models
   * - ``tutorial_bme_proba``
     - BMEPROBALIBtutorial
     - Full BME with soft probabilistic data
   * - ``tutorial_bme_interval``
     - BMEINTLIBtutorial
     - BME with interval data
   * - ``tutorial_kriging``
     - BMEHRLIBtutorial
     - Kriging (hard data only)
   * - ``tutorial_statistics``
     - STATLIBtutorial
     - Statistics, variograms, fitting
   * - ``tutorial_genlib``
     - GENLIBtutorial
     - Grid creation, NN, kernel smoothing

Run any tutorial from the command line::

    python -m pybme.tutorials.tutorial_models
    python -m pybme.tutorials.tutorial_bme_proba
    python -m pybme.tutorials.tutorial_bme_interval
    python -m pybme.tutorials.tutorial_kriging
    python -m pybme.tutorials.tutorial_statistics
    python -m pybme.tutorials.tutorial_genlib
"""
