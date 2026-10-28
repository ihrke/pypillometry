Command-line tools
==================

Three programs are installed together with the package. They give a quick
look at a recording without writing any Python code.

All of them read Eyelink EDF files and eyedata objects saved with
:func:`~pypillometry.write_pickle` or
:meth:`~pypillometry.GenericEyeData.write_file` (e.g., ``.pkl`` or
``.pypil``). The format is detected from the file content, not from the
extension. Instead of a local file, you can also give an HTTP(S) URL or an
SSH URL (``ssh://server/folder/file.edf``).

All three accept ``-q`` (only warnings and errors, no progress bars) and
``-v`` (debug messages, including the output of ``eyelinkio``). Use
``--help`` to see all options.


``pypilview``: look at a recording
----------------------------------

Opens a recording in the interactive :doc:`viewer </user_guide/viewer>`:

.. code-block:: bash

    pypilview recording.edf
    pypilview recording.edf --eyes left --variables pupil
    pypilview preprocessed.pypil

Press :kbd:`H` in the viewer for the keyboard controls.


``pypilinfo``: summarize files
------------------------------

Prints a summary of one or more files: variables, sampling rate, duration,
events, blinks and the processing history.

.. code-block:: bash

    pypilinfo recording.pkl
    pypilinfo data/*.pkl


``edf2pypil``: convert EDF files
--------------------------------

Reading large EDF files takes a while (about a minute for 90 MB).
``edf2pypil`` reads an EDF file once and saves it as a compressed pickle
file that loads much faster:

.. code-block:: bash

    edf2pypil recording.edf              # writes recording.pkl
    edf2pypil recording.edf -o rec.pypil
    edf2pypil recording.edf -f           # overwrite an existing file

Load the result in Python with :func:`~pypillometry.read_pickle` or
:meth:`EyeData.from_file() <pypillometry.GenericEyeData.from_file>`.


Eyelink triggers
----------------

By default, Eyelink's own messages (calibration, validation etc.) are
removed from the events when an EDF file is read. Pass ``--keep-triggers``
to keep them.
