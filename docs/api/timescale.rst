.. currentmodule:: pypillometry

Timescale model (experimental)
==============================

A state-space model of the pupil size with one fast component and slow
components at fixed time scales, for one recording
(:class:`TimescaleModel`) or a group of recordings with shrinkage
(:class:`TimescaleGroup`). See :doc:`/experimental/timescale_model` for an
introduction.

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   TimescaleModel
   TimescaleGroup
   timescale.plots.TimescalePlots
   timescale.plots.TimescaleGroupPlots

Functions on arrays
-------------------

The models are built from functions in ``pp.signal`` that work on plain
arrays.

.. currentmodule:: pypillometry.signal

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   block_mean
   coarsen
   semivariogram
   plot_variogram
   spectrum
   plot_spectrum

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Variogram
   Spectrum

Gauss-Markov models: ``pp.signal.gaussmarkov``
-----------------------------------------------

.. currentmodule:: pypillometry.signal.gaussmarkov

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   TimescaleBank
   GaussianPrior
   BankFit
   GroupFit
   VariogramCheck
   GaussMarkovModel
   Matern
   OU
   Constant
   LGSSM

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   fit_bank
   general_prior
   reference_prior
   fit_group
   blocked_cv
   variogram_check
   kalman_filter
   kalman_loglik
   rts_smoother
   forecast_errors
   simulate
