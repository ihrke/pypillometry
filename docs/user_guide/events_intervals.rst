.. currentmodule:: pypillometry

Events, intervals and segments
==============================

Most experiments send *events* (triggers) to the eye tracker, e.g., when a
stimulus appears or a response is given. Analyses then look at the data in
*intervals* around these events. :mod:`pypillometry` has a class for each
step:

.. mermaid::

   flowchart LR
     E["<b>Events</b><br/>onsets + labels"] -- "get_intervals()" --> I["<b>Intervals</b><br/>(start, end) pairs"]
     I -- "get_segments()" --> S["<b>SegmentedEyeData</b><br/>segments x time"]
     S -- "group_segments()" --> G["<b>GroupLevelSegmentedData</b><br/>participants x time"]
     I -- "stat_per_event()" --> N["one value<br/>per event"]


Events
------

:meth:`~GenericEyeData.get_events` returns the events of a dataset as an
:class:`Events` object, a list of onsets and labels:

.. code-block:: python

    events = d.get_events()              # onsets in ms
    events = d.get_events(units="sec")   # onsets in seconds
    events.filter("stim")                # labels containing "stim"
    events.filter(lambda lab: lab.endswith("_1"))
    events.filter((60, 120))             # onsets between 60 and 120 s
    events[:10]                          # the first ten events

:meth:`~Events.filter` and indexing return new :class:`Events`. You can
store changed events in the dataset with :meth:`~GenericEyeData.set_events`,
e.g., to remove test trials or to rename labels.

Tutorial: :doc:`/notebooks/events`.


Intervals
---------

An :class:`Intervals` object is a list of (start, end) pairs, together with
their units and, where applicable, the events they belong to. Intervals
come from several places:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Source
     - Example
   * - windows around events
     - ``d.get_intervals("stim", (-200, 1000) * u.ms)``
   * - spans between two events
     - ``d.get_intervals(("trial_start", "response"), (0, 0))``
   * - detected blinks
     - ``d.get_blinks("left", "pupil")``
   * - other detection methods
     - ``d.mask_offscreen_coords(apply_mask=False)``
   * - masked samples
     - ``d.mask_as_intervals("left", "pupil")``
   * - your own
     - ``pp.Intervals([(0, 100), (200, 300)] * u.ms)``

Intervals can be converted to other units (:meth:`~Intervals.to_units`),
filtered (:meth:`~Intervals.filter`), padded (:meth:`~Intervals.pad`),
merged when they overlap (:meth:`~Intervals.merge`), combined (``+``,
:func:`merge_intervals`), subtracted (``-``), summarized
(:meth:`~Intervals.stats`, :meth:`~Intervals.as_pandas`) and plotted
(:meth:`~Intervals.plot`).

Many methods of the data objects take intervals as input:

.. autosummary::
   :nosignatures:

   GenericEyeData.get_segments
   GenericEyeData.stat_per_event
   GenericEyeData.mask_intervals
   GenericEyeData.interpolate_intervals
   PupilPlotter.plot_intervals

Tutorial: :doc:`/notebooks/intervals`.


Segments and event-related pupil dilation
-----------------------------------------

:meth:`~GenericEyeData.get_segments` cuts the data of one variable in each
interval and returns a :class:`SegmentedEyeData` object (a matrix of
segments × time points). Averaged over segments, this is the
*event-related pupil dilation* (ERPD):

.. code-block:: python

    intervals = d.get_intervals("stim", (-200, 1000) * u.ms)
    erpd = d.get_segments(intervals, "left_pupil", name="stimulus")
    erpd = erpd.baseline_correct((-200, 0))   # subtract the pre-stimulus mean
    erpd.plot()

:func:`group_segments` averages the ERPDs of several participants into a
:class:`GroupLevelSegmentedData` object, which can be plotted the same way.
Both can be saved with ``write_file()``.

For statistical models of trial-by-trial data, it is often easier to work
with one number per trial. :meth:`~GenericEyeData.stat_per_event` applies a
function (by default the mean) to each interval:

.. code-block:: python

    peak = d.stat_per_event(interval=(500, 1500) * u.ms, event_select="stim",
                            variables="pupil", statfct=np.max)

Tutorial: :doc:`/notebooks/erpds`.
