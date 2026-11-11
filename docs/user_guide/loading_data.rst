.. currentmodule:: pypillometry

Loading and saving data
=======================

Eyelink EDF files
-----------------

EDF files from SR Research Eyelink eye trackers are read with
:meth:`~EyeData.from_eyelink`, which uses the `eyelinkio
<https://github.com/scott-huberty/eyelinkio>`_ package:

.. code-block:: python

    import pypillometry as pp

    d = pp.EyeData.from_eyelink("recording.edf")
    p = d.drop_data(variables=["x", "y"], inplace=False)   # pupil only

The result contains all samples, all messages (as events), the screen
resolution and the calibration/validation results (as
:class:`SpatialCalibration` objects in ``d.calibration``). Eyelink codes
missing pupil samples as 0; these are converted to NaN and masked.
Eyelink's own messages (calibration, drift correction, ...) are removed from
the events unless you pass ``remove_eyelink_triggers=False``.

Instead of a local file name, you can also pass an HTTP(S) URL or an SSH
URL (``ssh://server/folder/file.edf``). The file is then downloaded and
cached (see :func:`fetch_url`).

Files from public `figshare <https://figshare.com>`_ items are listed with
:func:`get_figshare_files` and downloaded (cached, checksum-verified) with
:func:`fetch_figshare_file`, using the item's ID, URL or DOI:

.. code-block:: python

    pp.get_figshare_files(28776617)    # file name -> id, url, size, md5
    fname = pp.fetch_figshare_file(28776617, "001_rlmw_samples.asc",
                                   cache_dir="data")

Reading large EDF files is slow (about a minute for a 90 MB file). Save
the result (see below), which is read in well under a second, or convert
the file once with the :doc:`command-line tool </getting_started/cli>`
``edf2pypil``.

See :doc:`/notebooks/importdata_edf`.


Tobii Pro Glasses (Tobii Pro Lab exports)
-----------------------------------------

Recordings of Tobii Pro Glasses 3 are read from the data export of Tobii
Pro Lab (an Excel or tab-separated file) with
:meth:`HeadMountedEyeData.from_tobii_prolab`:

.. code-block:: python

    d = pp.HeadMountedEyeData.from_tobii_prolab("rec.xlsx")   # and rec.mp4

    d = d.scene.luminance()                    # luminance of the scene video
    d.plot.plot_scene_scanpath((10, 12))       # gaze on a frame of the video
    d.scene.render_video("gaze.mp4", (10, 30)) # video with the gaze position

The result is a :class:`HeadMountedEyeData` object, an :class:`EyeData`
object with the additional data of eye-tracking glasses:

- ``left_pupil``, ``right_pupil``: the pupil diameter in mm
- ``both_x``, ``both_y``: the gaze position in the video of the scene
  camera (pixels; one position for both eyes)
- ``d.aux``: channels that do not belong to an eye, e.g. the gyroscope
  (``d.aux["gyro_x"]``), the accelerometer and the magnetometer,
  interpolated to the times of the eye-tracker samples, and the 3D gaze
  point
- the name of the scene video (the file with the name of the export and
  the extension ``.mp4``, or the argument ``video``)

The methods in ``d.scene`` add channels to ``d.aux`` that can be used as
covariates of the pupil size: the luminance of the scene video
(:meth:`~eyedata.head_mounted.SceneNamespace.luminance`), the viewing
distance (:meth:`~eyedata.head_mounted.SceneNamespace.distance`) and the
speed of the head rotation
(:meth:`~eyedata.head_mounted.SceneNamespace.head_movement`).
:meth:`~eyedata.head_mounted.SceneNamespace.render_video` writes the
scene video with the gaze position drawn on it. The channels in ``d.aux``
can be plotted with ``d.plot.plot_timeseries(variables=["pupil",
"luminance", "gyro"])`` and, against the pupil size, with
``d.plot.plot_pupil_covariate("gaze_luminance")``.

Excel files need the package `openpyxl`, the scene video the package `av`
(both are optional: ``pip install openpyxl av``). Reading the Excel file
is slow; save the result (see below).


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

Any of the eyes or variables can be left out (e.g., only ``left_pupil``
for pupil data without gaze); the x and y of an eye go together. Gaps in the
time vector (e.g., when recording was paused) are filled with masked
samples, so the samples are evenly spaced.

See :doc:`/notebooks/importdata_asc` for a complete example.

Events that are only in the behaviour log of the experiment (on its own
clock) can be added after loading; see
:ref:`behaviour-log-events`.


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
:meth:`~EyeData.write_file` and loaded with
:meth:`~EyeData.from_file`:

.. code-block:: python

    d.write_file("subject01.pypil")
    d = pp.EyeData.from_file("subject01.pypil")

Files are pickle files compressed with zstd (pass ``compress=False`` to
turn this off). :func:`write_file` and :func:`read_file` do the same
for any Python object, e.g., a list of objects or a
:class:`SegmentedEyeData`. :func:`read_file` also reads files from a URL.

.. warning::

   Pickle files can contain arbitrary code. Only load files from sources
   you trust.

If the filename ends in ``.h5`` or ``.hdf5``, the object is saved in an
HDF5 file instead:

.. code-block:: python

    d.write_file("subject01.h5")
    d = pp.EyeData.from_file("subject01.h5")

HDF5 files do not run code when they are read, and other programs (e.g.,
h5py, R or MATLAB) can read them: the data are datasets (``tx``,
``data/values/left_pupil``, ``data/layers/left_pupil``, ...; the mask layers
are bits, bit *i* is the layer ``data/layer_names[i]``, e.g. ``"blink"``), the
other information (name, sampling rate, events, history, ...) is stored in
groups and attributes. They work for eyedata objects and :class:`SegmentedEyeData`
(not for arbitrary Python objects). Values in ``info`` or ``params`` that
HDF5 cannot store are saved as text (with a warning). The loading
functions detect the format from the file content.

Both formats store a versioned description of the object. Files written
by older versions of pypillometry are upgraded when they are loaded;
``pypil-upgrade`` converts them once (see :doc:`/getting_started/cli`).

Saved files can be inspected with ``pypilinfo`` and opened in the viewer
with ``pypilview`` (see :doc:`/getting_started/cli`).


Whole studies
-------------

For studies with many participants, a small configuration file describes
where the raw data are and how to read them. :func:`load_study_local` and
:func:`load_study_osf` then load all participants at once (see
:doc:`studies`).
