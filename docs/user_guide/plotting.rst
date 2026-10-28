.. currentmodule:: pypillometry

Plotting
========

Every data object has a ``plot`` attribute with the plotting functions
that fit its data: a :class:`PupilPlotter` for :class:`PupilData`, a
:class:`GazePlotter` for :class:`GazeData` and an :class:`EyePlotter` (with
both sets of functions) for :class:`EyeData`. All plots use
:mod:`matplotlib`, so they can be adjusted and saved as usual. Most
functions take ``eyes=``, ``variables=``, ``plot_range=`` and ``units=``
arguments.

The examples on this page use the short example dataset with detected
blinks:

.. plot::
   :context: reset
   :nofigs:

   import pypillometry as pp
   from pypillometry import u
   import matplotlib.pyplot as plt

   d = pp.get_example_data("rlmw_002_short")
   d.pupil_blinks_detect(apply_mask=True)

Each function's API documentation has another example.


Time series
-----------

.. autosummary::
   :nosignatures:

   PupilPlotter.pupil_plot
   PupilPlotter.plot_timeseries
   PupilPlotter.plot_masked

:meth:`~PupilPlotter.pupil_plot` shows the pupil size of all eyes in one
plot, with blinks (red) and events (grey lines):

.. plot::
   :context: close-figs

   plt.figure(figsize=(10, 3))
   d.plot.pupil_plot(plot_range=(0, 20) * u.sec)

:meth:`~PupilPlotter.plot_timeseries` plots each variable (here: gaze and
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

   PupilPlotter.plot_timeseries_segments
   PupilPlotter.pupil_plot_segments


Blinks and intervals
--------------------

.. autosummary::
   :nosignatures:

   PupilPlotter.plot_blinks
   PupilPlotter.plot_intervals
   Intervals.plot
   Intervals.plot_highlight
   Events.plot

:meth:`~PupilPlotter.plot_blinks` draws one small plot per blink, e.g., to
check blink detection and interpolation:

.. plot::
   :context: close-figs

   d.plot.plot_blinks(eyes="left", nrow=3, ncol=5, figsize=(12, 6),
                      units="sec")

:meth:`~PupilPlotter.plot_intervals` does the same for any
:class:`Intervals`, here the three seconds around each stimulus:

.. plot::
   :context: close-figs

   intervals = d.get_intervals("S_", (-500, 2500) * u.ms)
   d.plot.plot_intervals(intervals, variables="pupil", nrow=2, ncol=5,
                         figsize=(12, 4))

:meth:`Intervals.plot_highlight` shades intervals on an existing plot:

.. plot::
   :context: close-figs

   intervals = d.get_intervals("S_", (0, 1000) * u.ms, units="sec")
   plt.figure(figsize=(10, 3))
   d.plot.pupil_plot(eyes="left", plot_range=(0, 20) * u.sec,
                     show_events=False)
   intervals.plot_highlight(color="green", alpha=0.2)
   plt.xlim(0, 20)


Event-related pupil dilation
----------------------------

The segments returned by :meth:`~GenericEyeData.get_segments` have their
own ``plot()`` method that shows the mean (with its standard error) and the
percentage of missing data (see :doc:`events_intervals`):

.. plot::
   :context: close-figs

   d.pupil_blinks_interpolate()
   intervals = d.get_intervals("S_", (-500, 2500) * u.ms)
   erpd = d.get_segments(intervals, "left_pupil").baseline_correct((-500, 0))
   erpd.plot()

.. autosummary::
   :nosignatures:

   SegmentedEyeData.plot
   GroupLevelSegmentedData.plot


Gaze
----

.. autosummary::
   :nosignatures:

   GazePlotter.plot_heatmap
   GazePlotter.plot_scanpath
   GazePlotter.plot_calibration

.. plot::
   :context: close-figs

   d.plot.plot_heatmap(eyes="left")
   d.plot.plot_scanpath(plot_range=(10, 12) * u.sec, eyes="left")

:meth:`~GazePlotter.plot_calibration` shows the calibration and validation
results of the eye tracker (see :doc:`/experimental/spatial_calibration`).
Regions of interest can be drawn on top of heatmaps and scan paths (see
:doc:`/experimental/gaze_rois`).


Comparing versions of the data
------------------------------

To compare, e.g., raw and filtered data, plot both into the same axes and
give each a ``label_prefix`` and a ``style``:

.. plot::
   :context: close-figs

   filtered = d.pupil_lowpass_filter(cutoff=4, inplace=False)

   plt.figure(figsize=(10, 3))
   d.plot.pupil_plot(eyes="left", plot_range=(5, 15) * u.sec,
                     label_prefix="raw", style={"color": "grey"})
   filtered.plot.pupil_plot(eyes="left", plot_range=(5, 15) * u.sec,
                            label_prefix="filtered")

See the tutorial :doc:`/notebooks/plotting` for more examples.
