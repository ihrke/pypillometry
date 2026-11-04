.. currentmodule:: pypillometry

Data objects
============

Data classes
------------

All data are held in one class, :class:`EyeData`. It holds whatever your
eye tracker recorded, for one or both eyes: pupil size, gaze position
(x, y), or both. The experimental setup (screen geometry, see
:class:`ExperimentalSetup`) and the calibration results can be stored with
any of them.

.. mermaid::

   classDiagram
    GenericEyeData <|-- EyeData
    EyeData <|-- FakeEyeData
    GenericEyeData: data (EyeDataDict)
    GenericEyeData: events, layers, history, experimental_setup, calibration
    GenericEyeData: get_intervals(), get_segments(), downsample(), ...
    EyeData: plot (EyePlotter)
    EyeData: pupil (PupilNamespace)
    EyeData: gaze (GazeNamespace)
    PupilNamespace: blinks_detect(), lowpass_filter(), ...
    PupilNamespace: correct_foreshortening(), ...
    GazeNamespace: mask_offscreen(), mask_divergences()

The generic methods (events, intervals, masks, downsampling, slicing, ...)
work for any data. The methods that need pupil data are in the namespace
``d.pupil`` (e.g., ``d.pupil.blinks_detect()``); they raise an error if
the object has no pupil data ("no pupil data; available channels: left_x,
left_y"). The same holds for the methods that need gaze data, in ``d.gaze``
(e.g., :meth:`EyeData.gaze.mask_offscreen`), or both (e.g.,
:doc:`foreshortening correction </experimental/foreshortening>`).

An object is usually created by reading a file (see
:doc:`loading_data`):

.. code-block:: python

    import pypillometry as pp
    d = pp.EyeData.from_eyelink("recording.edf")

Printing the object, or showing it in a notebook, gives an overview of
its contents. :meth:`~GenericEyeData.summary` returns the same information
as a dictionary.


Eyes and variables
------------------

The time series are stored in the attribute ``data``, an
:class:`EyeDataDict`. Its keys have the form ``<eye>_<variable>``, e.g.
``left_pupil``, ``right_x`` or ``left_y``. All time series have the same
length as the time vector ``d.tx`` (in ms), so they stay aligned.

.. code-block:: python

    d.eyes                 # ['left', 'right']
    d.variables            # ['x', 'y', 'pupil']
    d["left_pupil"]        # a numpy masked array
    d["left", "pupil"]     # the same
    d.tx                   # time in ms
    d.fs                   # sampling rate in Hz

Most methods take ``eyes=`` and ``variables=`` arguments. When they are
left empty, all eyes and variables are processed:

.. code-block:: python

    d.pupil.lowpass_filter(cutoff=4)                 # both eyes
    d.pupil.lowpass_filter(cutoff=4, eyes="left")    # only the left eye
    d.plot.plot_timeseries(variables=["x", "y"])     # only gaze

New eyes and variables can be added. For example,
:meth:`~EyeData.pupil.merge_eyes` combines the pupils of both eyes into
``mean_pupil`` (eye ``mean``, variable ``pupil``), and you can store
your own arrays with ``d["left", "myvar"] = array``.

See the tutorial :doc:`/notebooks/eyes_and_variables` for examples.


Masks
-----

Samples can be *masked*, i.e. marked as invalid without removing them.
Missing samples (NaN), blinks and samples where the gaze is off the
screen are typical examples. Every time series is a
:class:`numpy.ma.MaskedArray`, and masked samples are ignored by
averages, segments and statistics. Interpolated samples (e.g., blinks
after :meth:`~EyeData.pupil.blinks_interpolate`) are *not* masked by
default: they are used like other samples (see the mask policies below).

.. code-block:: python

    d.pupil.blinks_detect()                     # detect and mask blinks
    d.data.count_masked()                       # number of masked samples
    d.mask_as_intervals("left", "pupil")        # masked stretches (dict of Intervals)
    d.plot.plot_masked()                        # where is data masked?

Detection methods store what they find as a mask layer (e.g.,
``"blink"``), which masks the data, and return the object; read the
results with getters such as :meth:`~GenericEyeData.get_blinks` or
:meth:`~GenericEyeData.get_layer_intervals` (a dict keyed like
``"left_pupil"``, in ms) and undo them with
:meth:`~GenericEyeData.clear_layer`.

Each variable also keeps *mask layers* that record why a sample is
masked: ``"blink"`` (:meth:`~EyeData.pupil.blinks_detect`,
:meth:`~GenericEyeData.set_blinks`), ``"interpolated"``
(:meth:`~EyeData.pupil.blinks_interpolate`,
:meth:`~GenericEyeData.interpolate_intervals`), ``"offscreen"``
(:meth:`~EyeData.gaze.mask_offscreen`), ``"divergence"``
(:meth:`~EyeData.gaze.mask_divergences`) and ``"user"``
(:meth:`~GenericEyeData.mask_intervals`, assigning a masked array). The
``"excluded"`` layer (see below) belongs to the whole recording, not to
a variable. The blinks returned by :meth:`~GenericEyeData.get_blinks` are
the stretches of the ``"blink"`` layer. Running a detector again replaces
its layer (and its mask), and :meth:`~GenericEyeData.clear_layer` removes
a layer. The quality report splits the masked time by layer.
:meth:`~GenericEyeData.mask` and :meth:`~GenericEyeData.values` give the
mask (or the values with NaN) under a :class:`MaskPolicy`; by default,
interpolated samples count as usable:

.. code-block:: python

    d.mask("left", "pupil")                     # NaN, blinks, ...; not interpolated
    d.mask("left", "pupil", policy="strict")    # interpolated samples too
    d.values("left", "pupil")                   # values, NaN where masked
    d.clear_layer("blink")                      # undo blink detection

All processing and analysis methods read the data under one policy: the
default one unless you pass ``policy=`` (``"strict"`` leaves out the
interpolated samples, ``"raw"`` uses all values). Filters
(:meth:`~EyeData.pupil.lowpass_filter`,
:meth:`~EyeData.pupil.smooth_window`) and the baseline model bridge the
unusable samples linearly instead of using their values, and only change
the usable samples; masked samples keep their values. Segments,
:meth:`~GenericEyeData.stat_per_event`, :meth:`~GenericEyeData.scale`,
:meth:`~GenericEyeData.merge_eyes`, :meth:`~GenericEyeData.downsample` and
the quality report leave out the unusable samples. ``d["left_pupil"]`` and
``d.data.mask`` give the mask under the default policy.

.. code-block:: python

    seg = d.get_segments(epochs, "left_pupil", policy="strict")  # without interpolated samples
    d.stat_per_event(epochs, policy="strict")


Excluded periods
----------------

Some parts of a recording are not part of the analysis at all, e.g., the
calibration at the start or breaks between blocks. The data there may be
perfectly valid, so they should not count as missing data or blinks.
:meth:`~GenericEyeData.exclude_intervals` stores such periods with the
object (the recording-wide ``"excluded"`` layer) and masks all variables
during them, also variables added later:

.. code-block:: python

    breaks = d.get_intervals(("break_start", "break_end"), units="ms")
    d = d.exclude_intervals(breaks)
    d = d.exclude_intervals([(0, 2 * u.min)])   # calibration at the start
    d = d.exclude_intervals(d.get_intervals(("<start>", "Start Block 1"), units="ms"))
    d.get_excluded()                            # the excluded periods
    d.excluded_duration("min")                  # their total duration

:meth:`~GenericEyeData.keep_intervals` does the opposite: it excludes
everything outside the given periods, e.g., to keep only the fMRI run:

.. code-block:: python

    run = d.get_intervals(("rsfMRI_ONSET", "rsfMRI_END"), units="ms")
    d = d.keep_intervals(run)

Because the samples are masked, everything that ignores masked data
(averages, :meth:`~GenericEyeData.scale`, segments, ...) also ignores the
excluded periods (also where the data were interpolated). Unlike other
masks, excluded periods are not blinks: blink detection ignores them, and
blinks that overlap or touch an excluded period are not counted (they are
left out by :meth:`~GenericEyeData.get_blinks`), so they are not
interpolated either. The layer moves with the samples when the data are
sliced or downsampled, and it is saved with the object;
``d.clear_layer("excluded")`` removes the excluded periods.

:meth:`~GenericEyeData.get_included` returns the periods between them. To
get rid of the excluded data altogether, :meth:`~GenericEyeData.split_excluded`
returns one object per included period, and
:meth:`~GenericEyeData.remove_intervals` deletes the excluded samples and
joins the rest (the time vector and the events are shifted so that there
are no gaps):

.. code-block:: python

    d.get_included("sec")          # the periods that are not excluded
    parts = d.split_excluded()     # list of objects, original time
    joined = d.remove_intervals(d.get_excluded())   # one object, time re-aligned


Pipelines and history
---------------------

Processing steps can be combined into a pipeline with ``|``:

.. code-block:: python

    d = pp.EyeData.from_eyelink("recording.edf")
    d = d | pp.pupil.blinks_detect() \
          | pp.pupil.blinks_interpolate() \
          | pp.pupil.lowpass_filter(cutoff=4) \
          | pp.downsample(50)

``pp.pupil.<method>`` and ``pp.gaze.<method>`` are the methods of
``d.pupil`` and ``d.gaze``, and the generic methods are available at the
top level (``pp.crop``, ``pp.downsample``, ``pp.scale``, ...), with the
same arguments. ``d | step`` does not change ``d``: it returns a processed
copy. A pipeline is a value that can be stored and applied to several
datasets; applied as a whole, it copies the data only once:

.. code-block:: python

    preproc = pp.pupil.blinks_detect() | pp.pupil.blinks_interpolate() \
              | pp.pupil.lowpass_filter(cutoff=4)
    print(preproc)
    data = {subj: d | preproc for subj, d in datasets.items()}

The same methods can be called directly, e.g. ``d.pupil.lowpass_filter(4)``
or ``d.downsample(50)``; they return the object, so calls can be chained.

Every step is stored with its parameters in the object's history, shown at
the bottom of the object's summary and by
:meth:`~GenericEyeData.print_history`. This keeps a record of exactly how
the data were processed. See :doc:`/notebooks/pipes`.


In place or copy?
-----------------

By default, methods change the object they are called on (*in place*)
and return it. To keep the original, pass ``inplace=False`` (the method
then returns a changed copy) or copy the object first:

.. code-block:: python

    d2 = d.downsample(50, inplace=False)    # d is unchanged
    d2 = d.copy().downsample(50)            # the same

In notebooks, re-running a cell with an in-place operation applies it
again to the already processed data. If you prefer copies by default, set
``inplace=False`` when creating an object or for the whole package with
:func:`set_default_inplace` (see :doc:`settings`).


Fake data
---------

:class:`FakeEyeData` holds simulated data together with the true values
used to generate them, which is useful for testing preprocessing methods.
The functions for :ref:`simulated data <api-simulated-data>` generate the parts (slow
baseline changes, fixations, measurement noise), and
:func:`generate_foreshortening_data` creates a complete dataset (see
:doc:`/experimental/foreshortening`).
