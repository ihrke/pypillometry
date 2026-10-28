.. currentmodule:: pypillometry

API documentation
==================


.. automodule:: pypillometry


.. contents:: Table of Contents
    :local:
    :depth: 3


Overview 
--------

.. mermaid::

   classDiagram
    EyeData <|-- GazeData
    EyeData <|-- PupilData
    PupilData <|-- GenericEyeData
    GazeData <|-- GenericEyeData

    GazeData: +EyeDataDict data[left_x, left_y, right_x, right_y]
    GazeData: +GazePlotter plot
    GazeData: +set_experimental_setup() (and other gaze-specific functions)
    PupilData: +EyeDataDict data[left_pupil, right_pupil]
    PupilData: +PupilPlotter plot
    PupilData: +pupil_blinks_detect() (and other pupil-specific functions)
    EyeData: +EyeDataDict data[left_pupil, right_pupil, left_x, left_y, right_x, right_y]
    EyeData: +EyePlotter plot
    EyeData: +correct_pupil_foreshortening() (and other functions needing both gaze and pupil data)

.. autosummary::

    EyeData
    PupilData
    GazeData
    GenericEyeData
    EyeDataDict
    

Primary classes
---------------


Several different classes are available for handling
different types of eye data.

- :class:`PupilData`: If you have only pupillometric data, use this class.
- :class:`GazeData`: If you only have gaze data, use this class.
- :class:`EyeData`: If you have both gaze and pupillometric data, use this class.

These classes inherit from :class:`GenericEyeData`, which provides some basic, shared functionality 
for handling eye data.

The data inside each of these classes is stored in the `data` attribute, which is a dictionary-like object
of type :class:`EyeDataDict`. 

Each of these classes provides access to a matching plotter object (from module :mod:`pypillometry.plot`) 
that is stored in the `plot` attribute.

EyeData 
^^^^^^^

.. autoclass:: pypillometry.EyeData
    :members:

PupilData
^^^^^^^^^

.. autoclass:: pypillometry.PupilData
    :members:

GazeData
^^^^^^^^

.. autoclass:: pypillometry.GazeData
    :members:

Plotting
--------

.. autoclass:: pypillometry.EyePlotter
    :members:
    :inherited-members:

.. autoclass:: pypillometry.PupilPlotter
    :members:
    :inherited-members:

.. autoclass:: pypillometry.GazePlotter
    :members:
    :inherited-members:


Intervals and events
--------------------

:class:`Intervals` hold time intervals (e.g., blinks or windows around
events, see :meth:`~pypillometry.GenericEyeData.get_intervals`);
:class:`Events` hold event onsets and labels
(see :meth:`~pypillometry.GenericEyeData.get_events`).

.. autoclass:: pypillometry.Intervals
    :members:

.. autofunction:: pypillometry.merge_intervals

.. autoclass:: pypillometry.Events
    :members:


Segmented data
--------------

Data cut into segments around events
(see :meth:`~pypillometry.GenericEyeData.get_segments`), for single
subjects and on the group level.

.. autoclass:: pypillometry.SegmentedEyeData
    :members:

.. autoclass:: pypillometry.GroupLevelSegmentedData
    :members:

.. autofunction:: pypillometry.group_segments


Regions of interest
-------------------

.. autoclass:: pypillometry.ROI
    :members:

.. autoclass:: pypillometry.CircularROI
    :members:

.. autoclass:: pypillometry.RectangularROI
    :members:


Supporting classes
------------------

These classes are used internally by the main classes will only rarely be used directly by the user.



GenericEyeData
^^^^^^^^^^^^^^

.. autoclass:: pypillometry.GenericEyeData
    :members:

EyeDataDict
^^^^^^^^^^^

.. autoclass:: pypillometry.EyeDataDict
    :members:

Stateless signal-processing functions
--------------------------------------

These functions are used by the main classes to process the data. They are stateless, meaning that they do not store any state between calls.

.. automodule:: pypillometry.signal.baseline
    :members:

.. automodule:: pypillometry.signal.preproc
    :members:

.. automodule:: pypillometry.signal.pupil
    :members:

.. automodule:: pypillometry.signal.fake
    :members:


Units
-----

Times, distances and angles can be given with units, either multiplied by a
unit from :data:`pypillometry.u` or as strings:

.. code-block:: python

    from pypillometry import u

    d.sub_slice(3*u.min, 5*u.min)                   # or ("3 min", "5 min")
    d.get_intervals("stim", (-200, 800) * u.ms)
    d.pupil_blinks_detect(min_duration="20 ms")
    d.plot.pupil_plot(plot_range=(3, 5) * u.min)
    ExperimentalSetup(eye_to_screen_center=60 * u.cm, screen_pitch="-5 deg")

Internally, all values are converted to ms, mm and radians. Plain numbers
are interpreted as documented for each function (usually ms for times).
Available units: ``u.ms``, ``u.sec``, ``u.min``, ``u.h``, ``u.samples``,
``u.mm``, ``u.cm``, ``u.m``, ``u.inch``, ``u.deg``, ``u.rad`` (plus aliases
such as ``u.seconds`` or ``u.degree``).

.. autoclass:: pypillometry.Quantity
    :members: to

.. autofunction:: pypillometry.parse_time

.. autofunction:: pypillometry.parse_distance

.. autofunction:: pypillometry.parse_angle


Logging
-------

By default, the log-level is set to `INFO` which results in a moderate amount of logging information.
The logging can be turned off by running :func:`~pypillometry.logging_disable` and turned back on by running :func:`~pypillometry.logging_enable`.
You can also set the log-level to `DEBUG` or `WARN` by running :func:`~pypillometry.logging_set_level`.

.. autofunction:: pypillometry.logging_set_level

.. autofunction:: pypillometry.logging_disable

.. autofunction:: pypillometry.logging_enable



Modules
-------


.. automodule:: pypillometry.io
    :members: osf_authenticate, read_eyelink, read_pickle, write_pickle, fetch_url, local_copy

.. automodule:: pypillometry.study
    :members: load_study_osf, load_study_local


.. automodule:: pypillometry.example_data
    :members: get_example_data, get_available_datasets

..
    .. automodule:: pypillometry.pupildata
        :members:
        :special-members:

    .. automodule:: pypillometry.erpd
        :members:

    .. automodule:: pypillometry.io
        :members:

    .. automodule:: pypillometry.preproc
        :members:

    .. automodule:: pypillometry.fakedata
        :members:

    .. automodule:: pypillometry.baseline
        :members:

    .. automodule:: pypillometry.pupil
        :members:

    .. automodule:: pypillometry.convenience
        :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`