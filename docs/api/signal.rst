.. currentmodule:: pypillometry

Signal processing
=================

These functions work on plain :mod:`numpy` arrays and do not keep any
state. The methods of the data classes call them with the right arguments,
so you usually do not need them directly, but they are useful for your
own processing steps or for data that are not in a data object.

Filtering and baseline
----------------------

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   butter_lowpass
   butter_lowpass_filter
   lowpass_filter_iterative
   smooth_window
   downsample
   bspline
   baseline_envelope
   baseline_envelope_iter_bspline
   baseline_pupil_model
   zscale

Blinks
------

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   velocity_savgol
   detect_blinks_velocity
   detect_blinks_zero
   blink_onsets_mahot

Pupil response
--------------

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   pupil_kernel
   pupil_kernel_t
   pupil_get_max_duration
   plot_prf
   pupil_build_design_matrix
   pupil_response
   pupilresponse_nnls
   pupil_signal_quality
   pupil_eye_offset

.. _api-simulated-data:

Simulated data
--------------

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   fake_pupil_baseline
   fake_gaze_fixations
   add_measurement_noise
   generate_foreshortening_data
