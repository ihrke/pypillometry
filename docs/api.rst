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


Logging
-------

By default, the log-level is set to `INFO` which results in a moderate amount of logging information.
The logging can be turned off by running :func:`pypillometry.logging_disable` and turned back on by running :func:`pypillometry.logging_enable`.
You can also set the log-level to `DEBUG` or `WARN` by running :func:`pypillometry.logging_set_level`.

.. autofunction:: pypillometry.logging_set_level

.. autofunction:: pypillometry.logging_disable

.. autofunction:: pypillometry.logging_enable

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