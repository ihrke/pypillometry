.. currentmodule:: pypillometry

Plotting
========

Every data object has a ``plot`` namespace with the plotting functions
(:class:`~pypillometry.plot.PlotNamespace`), e.g. ``d.plot.pupil_plot()``;
``dir(d.plot)`` lists them. Pupil plots need pupil data and gaze plots need
gaze data; they raise an error if the data are missing. Gaze plots draw the
screen as it is seen: pixel y increases downward. All plots use
:mod:`matplotlib`, so they can be adjusted and saved as usual. Most
functions take ``eyes=``, ``variables=``, ``plot_range=`` and ``units=``
arguments. ``units=`` is the unit of the time axis and of plain numbers
in ``plot_range`` (and ``interv`` of the ``*_segments`` plots):
``plot_range=(0, 20), units="sec"`` plots the first 20 seconds. Values
with units are converted, e.g., ``plot_range=(0, 20) * u.sec``.

The examples on this page use the short example dataset with detected
blinks:

.. plot::
   :context: reset
   :nofigs:

   import pypillometry as pp
   from pypillometry import u
   import matplotlib.pyplot as plt

   d = pp.get_example_data("rlmw_002_short")
   d.pupil.blinks_detect()

Each function's API documentation has another example.


Time series
-----------

.. autosummary::
   :nosignatures:

   EyeData.plot.pupil_plot
   EyeData.plot.plot_timeseries
   EyeData.plot.plot_masked

:meth:`~EyeData.plot.pupil_plot` shows the pupil size of all eyes in one
plot, with blinks (red) and events (grey lines):

.. plot::
   :context: close-figs

   plt.figure(figsize=(10, 3))
   d.plot.pupil_plot(plot_range=(0, 20) * u.sec)

:meth:`~EyeData.plot.plot_timeseries` plots each variable (here: gaze and
pupil) in its own panel:

.. plot::
   :context: close-figs

   plt.figure(figsize=(10, 6))
   d.plot.plot_timeseries(plot_range=(0, 20) * u.sec, units="sec")

To save a plot, use matplotlib as usual, e.g.
``plt.savefig("pupil.pdf")``.


Long recordings
---------------

Plotting a whole recording at once hides the details. These functions split
it into many small plots (one figure each), optionally written to a
multi-page PDF file with ``pdf_file=``. For interactive browsing, the
:doc:`viewer <viewer>` is usually faster.

.. autosummary::
   :nosignatures:

   EyeData.plot.plot_timeseries_segments
   EyeData.plot.pupil_plot_segments


Blinks and intervals
--------------------

.. autosummary::
   :nosignatures:

   EyeData.plot.plot_blinks
   EyeData.plot.plot_intervals
   Intervals.plot
   Intervals.plot_highlight
   Events.plot

:meth:`~EyeData.plot.plot_blinks` draws one small plot per blink, e.g., to
check blink detection and interpolation:

.. plot::
   :context: close-figs

   d.plot.plot_blinks(eyes="left", nrow=3, ncol=5, figsize=(12, 6),
                      units="sec")

:meth:`~EyeData.plot.plot_intervals` does the same for any
:class:`Intervals`, here the three seconds around each stimulus:

.. plot::
   :context: close-figs

   epochs = d.get_epochs("S_", (-500, 2500) * u.ms)
   d.plot.plot_intervals(epochs, variables="pupil", nrow=2, ncol=5,
                         figsize=(12, 4))

:meth:`Intervals.plot_highlight` shades intervals on an existing plot:

.. plot::
   :context: close-figs

   intervals = d.get_epochs("S_", (0, 1000) * u.ms).to_intervals("sec")
   plt.figure(figsize=(10, 3))
   d.plot.pupil_plot(eyes="left", plot_range=(0, 20) * u.sec,
                     show_events=False)
   intervals.plot_highlight(color="green", alpha=0.2)
   plt.xlim(0, 20)


Event-related pupil dilation
----------------------------

The segments returned by :meth:`~EyeData.get_segments` have their
own ``plot()`` method that shows the mean (with its standard error) and the
percentage of missing data (see :doc:`events_intervals`):

.. plot::
   :context: close-figs

   d.pupil.blinks_interpolate()
   epochs = d.get_epochs("S_", (-500, 2500) * u.ms)
   erpd = d.get_segments(epochs, "left_pupil").baseline_correct((-500, 0))
   erpd.plot()

.. autosummary::
   :nosignatures:

   SegmentedEyeData.plot
   GroupLevelSegmentedData.plot


Gaze
----

.. autosummary::
   :nosignatures:

   EyeData.plot.plot_heatmap
   EyeData.plot.plot_scanpath
   EyeData.plot.plot_calibration

.. plot::
   :context: close-figs

   d.plot.plot_heatmap(eyes="left")
   d.plot.plot_scanpath(plot_range=(10, 12) * u.sec, eyes="left")

:meth:`~EyeData.plot.plot_calibration` shows the calibration and validation
results of the eye tracker (see :doc:`/experimental/spatial_calibration`).
Regions of interest can be drawn on top of heatmaps and scan paths (see
:doc:`/experimental/gaze_rois`).


Comparing versions of the data
------------------------------

To compare, e.g., raw and filtered data, plot both into the same axes and
give each a ``label_prefix`` and a ``style``:

.. plot::
   :context: close-figs

   filtered = d | pp.pupil.lowpass_filter(cutoff=4)

   plt.figure(figsize=(10, 3))
   d.plot.pupil_plot(eyes="left", plot_range=(5, 15) * u.sec,
                     label_prefix="raw", style={"color": "grey"})
   filtered.plot.pupil_plot(eyes="left", plot_range=(5, 15) * u.sec,
                            label_prefix="filtered")

See the tutorial :doc:`/notebooks/plotting` for more examples.
