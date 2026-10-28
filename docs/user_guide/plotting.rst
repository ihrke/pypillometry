.. currentmodule:: pypillometry

Plotting
========

Every data object has a ``plot`` attribute with the plotting functions
that fit its data: a :class:`PupilPlotter` for :class:`PupilData`, a
:class:`GazePlotter` for :class:`GazeData` and an :class:`EyePlotter` (with
both sets of functions) for :class:`EyeData`. All plots use
:mod:`matplotlib`, so they can be adjusted and saved as usual.

.. code-block:: python

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    d.plot.pupil_plot(plot_range=(2, 3) * u.min, eyes="left")
    plt.savefig("pupil.pdf")

Most functions take ``eyes=``, ``variables=``, ``plot_range=`` and
``units=`` arguments.

Time series
-----------

.. autosummary::
   :nosignatures:

   PupilPlotter.plot_timeseries
   PupilPlotter.pupil_plot
   PupilPlotter.plot_masked

Long recordings
---------------

Plotting a whole recording at once hides the details. These functions split
it into many small plots, optionally written to a multi-page PDF file. For
interactive browsing, the :doc:`viewer <viewer>` is usually faster.

.. autosummary::
   :nosignatures:

   PupilPlotter.plot_timeseries_segments
   PupilPlotter.pupil_plot_segments

Blinks and intervals
--------------------

One small plot per blink or interval, e.g., to check blink detection and
interpolation:

.. autosummary::
   :nosignatures:

   PupilPlotter.plot_blinks
   PupilPlotter.plot_intervals
   Intervals.plot
   Events.plot

Gaze
----

.. autosummary::
   :nosignatures:

   GazePlotter.plot_heatmap
   GazePlotter.plot_scanpath
   GazePlotter.plot_calibration

Comparing versions of the data
------------------------------

To compare, e.g., raw and filtered data, plot both into the same axes and
give each a ``label_prefix`` and a ``style``:

.. code-block:: python

    raw = d.copy()
    d.pupil_lowpass_filter(cutoff=4)

    raw.plot.pupil_plot(plot_range=(60, 70), label_prefix="raw",
                        style={"color": "grey"})
    d.plot.pupil_plot(plot_range=(60, 70), label_prefix="filtered")

See the tutorial :doc:`/notebooks/plotting`.
