.. currentmodule:: pypillometry

Data objects
============

Data classes
------------

All data are held in one of three classes. Choose the one that matches
what your eye tracker recorded:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Class
     - Contains
     - Use for
   * - :class:`PupilData`
     - pupil size
     - pupillometry without gaze position
   * - :class:`GazeData`
     - gaze position (x, y)
     - eye movements without pupil size
   * - :class:`EyeData`
     - pupil size and gaze position
     - the usual case; needed for anything that combines both, e.g.,
       :doc:`foreshortening correction </experimental/foreshortening>`

.. mermaid::

   classDiagram
    GenericEyeData <|-- PupilData
    GenericEyeData <|-- GazeData
    PupilData <|-- EyeData
    GazeData <|-- EyeData
    GenericEyeData: data (EyeDataDict)
    GenericEyeData: events, blinks, history
    GenericEyeData: get_intervals(), get_segments(), downsample(), ...
    PupilData: plot (PupilPlotter)
    PupilData: pupil_blinks_detect(), pupil_lowpass_filter(), ...
    GazeData: plot (GazePlotter)
    GazeData: mask_offscreen_coords(), experimental_setup, ...
    EyeData: plot (EyePlotter)
    EyeData: correct_pupil_foreshortening(), ...

All three share the methods of :class:`GenericEyeData` (events, intervals,
masks, downsampling, slicing, ...). :class:`EyeData` has the methods of
both :class:`PupilData` and :class:`GazeData`. Methods that only make sense
for pupil data start with ``pupil_``.

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

    d.pupil_lowpass_filter(cutoff=4)                 # both eyes
    d.pupil_lowpass_filter(cutoff=4, eyes="left")    # only the left eye
    d.plot.plot_timeseries(variables=["x", "y"])     # only gaze

New eyes and variables can be added. For example,
:meth:`~PupilData.pupil_merge_eyes` combines the pupils of both eyes into
``merged_pupil`` (eye ``merged``, variable ``pupil``), and you can store
your own arrays with ``d["left", "myvar"] = array``.

See the tutorial :doc:`/notebooks/eyes_and_variables` for examples.


Masks
-----

Samples can be *masked*, i.e. marked as invalid without removing them.
Blinks, samples where the gaze is off the screen, or interpolated
segments are typical examples. Every time series is a
:class:`numpy.ma.MaskedArray`, and masked samples are ignored by
averages, segments and statistics.

.. code-block:: python

    d.pupil_blinks_detect(apply_mask=True)      # mask blinks
    d.data.count_masked()                       # number of masked samples
    d.mask_as_intervals("left", "pupil")        # masked stretches as Intervals
    d.plot.plot_masked()                        # where is data masked?

Detection methods return :class:`Intervals` (``apply_mask=False``, the
default) so that you can check them before applying them with
:meth:`~GenericEyeData.mask_intervals` (see :doc:`events_intervals`).


Pipelines and history
---------------------

Processing methods return the object itself, so several steps can be
chained:

.. code-block:: python

    d = pp.EyeData.from_eyelink("recording.edf")\
          .pupil_blinks_detect(apply_mask=True)\
          .pupil_blinks_interpolate()\
          .pupil_lowpass_filter(cutoff=4)\
          .downsample(50)

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
