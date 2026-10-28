.. currentmodule:: pypillometry

Loading and saving data
=======================

Eyelink EDF files
-----------------

EDF files from SR Research Eyelink eye trackers are read with
:meth:`~GenericEyeData.from_eyelink`, which uses the `eyelinkio
<https://github.com/scott-huberty/eyelinkio>`_ package:

.. code-block:: python

    import pypillometry as pp

    d = pp.EyeData.from_eyelink("recording.edf")
    p = pp.PupilData.from_eyelink("recording.edf")   # pupil only
    g = pp.GazeData.from_eyelink("recording.edf")    # gaze only

The result contains all samples, all messages (as events), the screen
resolution and the calibration/validation results (as
:class:`SpatialCalibration` objects in ``d.calibration``). Eyelink codes
missing pupil samples as 0; these are converted to NaN and masked.
Eyelink's own messages (calibration, drift correction, ...) are removed from
the events unless you pass ``remove_eyelink_triggers=False``.

Instead of a local file name, you can also pass an HTTP(S) URL or an SSH
URL (``ssh://server/folder/file.edf``). The file is then downloaded and
cached (see :func:`fetch_url`).

Reading large EDF files is slow. Save the result (see below) or convert
the file once with the :doc:`command-line tool </getting_started/cli>`
``edf2pypil``.

See :doc:`/notebooks/importdata_edf`.


Text files and your own arrays
------------------------------

Data from other eye trackers, or exported as text (e.g., Eyelink's ASC
format), can be read with :mod:`numpy` or :mod:`pandas` and passed to the
constructors as arrays:

.. code-block:: python

    d = pp.EyeData(
        time=df["time"],                      # in ms
        left_x=df["lx"], left_y=df["ly"], left_pupil=df["lp"],
        right_x=df["rx"], right_y=df["ry"], right_pupil=df["rp"],
        event_onsets=events["time"],          # in ms
        event_labels=events["label"],
        sampling_rate=500,
        name="subject01",
    )

Any of the eyes or variables can be left out. :class:`PupilData` and
:class:`GazeData` take the corresponding subset of arguments. Gaps in the
time vector (e.g., when recording was paused) are filled with masked
samples, so the samples are evenly spaced.

See :doc:`/notebooks/importdata_asc` for a complete example.


Example datasets
----------------

A few datasets are available for trying out the package:

.. code-block:: python

    pp.get_available_datasets()                # names and descriptions
    d = pp.get_example_data("rlmw_002_short")  # 40 s, included in the package
    d = pp.get_example_data("rlmw_002")        # 20 min, downloaded from OSF

See :doc:`/notebooks/example_datasets`.


Saving and loading objects
--------------------------

Processed objects, including their history, are saved with
:meth:`~GenericEyeData.write_file` and loaded with
:meth:`~GenericEyeData.from_file`:

.. code-block:: python

    d.write_file("subject01.pypil")
    d = pp.EyeData.from_file("subject01.pypil")

Files are pickle files compressed with zstd (pass ``compress=False`` to
turn this off). :func:`write_pickle` and :func:`read_pickle` do the same
for any Python object, e.g., a list of objects or a
:class:`SegmentedEyeData`. :func:`read_pickle` also reads files from a URL.

.. warning::

   Pickle files can contain arbitrary code. Only load files from sources
   you trust.

Saved files can be inspected with ``pypilinfo`` and opened in the viewer
with ``pypilview`` (see :doc:`/getting_started/cli`).


Whole studies
-------------

For studies with many participants, a small configuration file describes
where the raw data are and how to read them. :func:`load_study_local` and
:func:`load_study_osf` then load all participants at once (see
:doc:`studies`).
