.. currentmodule:: pypillometry

Pipelines
=========

Processing steps are combined with ``|``::

    d = d | (pp.pupil.blinks_detect() | pp.pupil.lowpass_filter(5) | pp.downsample(50))

``d | step`` returns a processed copy of ``d``. Steps combine into a
:class:`Pipeline`, a value that can be applied to several datasets and
copies the data only once (``d | a | b`` copies at each ``|``); see
:ref:`data-model-pipelines`.

Steps for pupil and gaze data
-----------------------------

``pp.pupil.<method>(...)`` and ``pp.gaze.<method>(...)`` make steps from the
methods of ``d.pupil`` and ``d.gaze``, with the same arguments (see
:class:`~eyedata.pupil.PupilNamespace` and
:class:`~eyedata.gaze.GazeNamespace`).

Generic steps
-------------

Each of these makes a step from the method of the same name
(e.g., ``pp.crop(...)`` calls :meth:`EyeData.crop`).

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   crop
   downsample
   scale
   unscale
   merge_eyes
   reset_time
   mask_intervals
   interpolate_intervals
   exclude_intervals
   keep_intervals
   remove_intervals
   merge_masks
   clear_layer
   set_blinks
   set_events
   add_events
   set_experimental_setup
   drop_data
   drop_original
   fill_time_discontinuities

Steps and pipelines
-------------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   Step
   Pipeline
