.. currentmodule:: pypillometry

Plotting (``d.plot``)
=====================

Each data object has a ``plot`` namespace with the plotting functions, e.g.
``d.plot.pupil_plot()`` (see :doc:`/user_guide/plotting`). You do not create
it yourself. The plots draw into the current matplotlib figure (or create
one).

Time series
-----------

.. autosummary::
   :nosignatures:

   plot.PlotNamespace.pupil_plot
   plot.PlotNamespace.plot_timeseries
   plot.PlotNamespace.pupil_plot_segments
   plot.PlotNamespace.plot_timeseries_segments

Blinks, masks and intervals
---------------------------

.. autosummary::
   :nosignatures:

   plot.PlotNamespace.plot_blinks
   plot.PlotNamespace.plot_masked
   plot.PlotNamespace.plot_intervals

Gaze
----

.. autosummary::
   :nosignatures:

   plot.PlotNamespace.plot_heatmap
   plot.PlotNamespace.plot_scanpath
   plot.PlotNamespace.plot_calibration

Experimental setup and foreshortening
-------------------------------------

.. autosummary::
   :nosignatures:

   plot.PlotNamespace.plot_experimental_setup
   plot.PlotNamespace.plot_pupil_foreshortening_error_surface

The segmented data have their own ``plot()`` method
(:meth:`SegmentedEyeData.plot`).

All plots
---------

.. autoclass:: pypillometry.plot.PlotNamespace
   :members:
   :inherited-members:
