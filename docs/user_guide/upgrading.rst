.. currentmodule:: pypillometry

Upgrading from 2.1
==================

Version 2.2 changes how the data are stored and how the methods are
called. Most scripts written for 2.1 need some changes. This page lists
what changed and what to look out for; the
:doc:`changelog </development/changelog>` has the complete tables of old
and new calls.


What changed
------------

**One data class.** :class:`EyeData` holds pupil and gaze data alike;
``PupilData``, ``GazeData`` and their base class ``GenericEyeData`` are
gone. Use
``d.drop_data(variables=["x", "y"], inplace=False)`` instead of
``d.to_pupildata()``. See :doc:`data_model`.

**Namespaces.** The methods for pupil data are in ``d.pupil``, those for
gaze data in ``d.gaze`` and the plots in ``d.plot``, without the prefixes:

.. code-block:: python

    d.pupil_blinks_detect()      # 2.1
    d.pupil.blinks_detect()      # 2.2

    d.mask_offscreen_coords()    # 2.1
    d.gaze.mask_offscreen()      # 2.2

    d.pupil_plot()               # 2.1
    d.plot.pupil_plot()          # 2.2

The old names are gone (an ``AttributeError``); the tables in the
changelog list the new name for every changed call.

**Pipelines.** Processing steps can be combined with ``|``
(``d | pp.pupil.blinks_detect() | pp.pupil.lowpass_filter(4)``); see
:ref:`pipelines <data-model-pipelines>`.

**Mask layers and policies.** Each channel keeps named mask layers
(``"blink"``, ``"interpolated"``, ...) instead of a single mask. Detectors
set their layer and return the object; the ``apply_mask`` argument is
gone, and the results are read with getters (``d.get_blinks()``,
``d.get_layer_intervals("offscreen")``). A mask policy decides which layers
count as unusable; by default, **interpolated samples are used**. See
:ref:`masks <data-model-masks>`.

**Events, intervals and epochs.** The events are one :class:`Events`
object (``d.events.onsets`` instead of ``d.event_onsets``).
Windows around events come from the new :meth:`~EyeData.get_epochs`
(``d.get_epochs("stim", window)`` returns :class:`Epochs`: event onsets
with one window); :meth:`~EyeData.get_intervals` only gives the spans
between two events and takes them as keywords
(``d.get_intervals(start="trial", end="response")``).
``pp.merge_intervals()`` is now :func:`concat_intervals` (it concatenates;
:meth:`Intervals.merge` merges overlapping intervals). See
:doc:`events_intervals`.

**Segments with several channels.** :meth:`~EyeData.get_segments` cuts
all channels by default (or those named); a channel is read with
``seg["left_pupil"]`` instead of ``seg.data``. As for the data, the pupil
methods are in ``seg.pupil``; the gaze methods (regions of interest) are
new in ``seg.gaze``:

.. code-block:: python

    erpd = d.get_segments(epochs, "left_pupil").baseline_correct((-200, 0))   # 2.1
    erpd.plot()                                                               # 2.1
    erpd = d.get_segments(epochs, "left_pupil").pupil.baseline_correct((-200, 0))  # 2.2
    erpd.pupil.plot()                                                         # 2.2

**No GroupLevelSegmentedData.** Averages are :class:`SegmentedEyeData`
objects whose columns are averages (see :doc:`events_intervals`).
:func:`group_segments` returns one with one column per participant; the
participants are in the metadata of its epochs instead of ``subjects``
and ``n_subjects``:

.. code-block:: python

    group.subjects                                   # 2.1
    group.epochs.meta["subject"]                     # 2.2


Changes that do not raise an error
----------------------------------

These changes silently give other results with code that ran under 2.1.
Check your scripts for them:

- **Plain numbers are ms** everywhere: windows of
  :meth:`~EyeData.get_epochs` and :meth:`~EyeData.get_intervals`,
  durations, margins. Sample indices
  need ``units="samples"``. Exception: in plots, plain numbers in
  ``plot_range`` and ``interv`` are in ``units=``, as in 2.1.
- **Interpolated samples are used** in segments,
  :meth:`~EyeData.stat_per_event`, :meth:`~EyeData.scale`,
  :meth:`~EyeData.merge_eyes`, :meth:`~EyeData.downsample` and
  the quality report. Pass ``policy="strict"`` for the 2.1 behaviour.
- **In place by default** also for :meth:`~EyeData.drop_data`,
  :meth:`~EyeData.fill_time_discontinuities` and
  ``seg.pupil.baseline_correct()``, which returned a copy in 2.1:
  ``d2 = d.drop_data(...)`` now changes ``d`` too. Pass ``inplace=False``.
- **Results per channel are dicts** keyed like the data (``"left_pupil"``),
  e.g. ``d.get_blinks()["left_pupil"]``, in ms.
- **Merged eyes** are called ``mean`` (``mean_pupil``), not ``merged``.
- **The sampling rate after** :meth:`~EyeData.downsample` is the
  actual rate: downsampling 1000 Hz data to 300 Hz gives 333.3 Hz.
- ``pp.downsample`` is now a pipeline step; the function for numpy arrays
  is ``pp.signal.downsample``. All functions for numpy arrays are only in
  ``pp.signal`` (``pp.signal.smooth_window``, ``pp.signal.pupil_kernel``,
  ``pp.signal.zscale``, ...).

All numbers that differ from 2.1, with the reason for each, are listed in
`tests/data/reference/CHANGES.md
<https://github.com/ihrke/pypillometry/blob/master/tests/data/reference/CHANGES.md>`_.


Saved files
-----------

Files saved with earlier versions (pickle and HDF5) can still be read:
they are upgraded each time they are loaded, and ``PupilData`` and
``GazeData`` objects load as :class:`EyeData`. ``pypil-upgrade`` converts
them once, so that this is not needed any more (see
:doc:`/getting_started/cli`):

.. code-block:: bash

    pypil-upgrade -n data/        # report what would be converted
    pypil-upgrade -i data/        # convert in place

What happens to the contents of old files:

- Blinks, interpolated blinks and excluded periods become the layers
  ``"blink"``, ``"interpolated"`` and ``"excluded"``. Stored blinks always
  mask the data, also blinks that were stored without
  ``apply_mask=True``.
- Other masked samples (e.g., from ``mask_offscreen_coords``,
  ``mask_eye_divergences`` or masks set by hand) become the ``"user"``
  layer: their origin is not stored in old files.
- A sampling rate that does not fit the time vector (files saved after
  downsampling to a rate that does not divide the original one) is
  corrected from the time vector, with a warning.
- Segments with event-locked intervals get :class:`Epochs`.
- Segments hold their one variable as a channel (``seg["left_pupil"]``).
- Groups (``GroupLevelSegmentedData``) load as :class:`SegmentedEyeData`.
  Averages of participants get the participants "1", "2", ... (metadata
  ``subject``) and an unknown number of trials (``n_trials`` is NaN);
  stacked trials do not know their participant.
