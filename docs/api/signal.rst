.. currentmodule:: pypillometry.signal

Signal processing
=================

These functions work on plain :mod:`numpy` arrays and do not keep any
state. They are in the submodule ``pp.signal`` (e.g.,
``pp.signal.smooth_window``). The methods of the data classes call them with the right arguments,
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
   zscale
   bspline
   baseline_envelope
   baseline_envelope_iter_bspline
   baseline_pupil_model

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
   interpolate_blinks_mahot
   interpolate_blinks_kalman

.. _api-transplant:

Transplantation of blinks
-------------------------

Real blinks are cut out and added to clean stretches of a signal, where
the signal without the blink is known (see
:meth:`d.pupil.blinks_transplant() <pypillometry.eyedata.pupil.PupilNamespace.blinks_transplant>`).

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   transplant_blinks
   transplant_error
   blink_cores
   blink_donors
   extract_blink_artifacts
   blink_artifact_profile
   shrink_blink_artifacts
   clean_stretches
   place_blink_artifacts
   insert_blink_artifacts

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BlinkTransplant

The parameters of the interpolation of blinks can be tuned on
transplanted blinks (see
:meth:`d.pupil.blinks_autotune() <pypillometry.eyedata.pupil.PupilNamespace.blinks_autotune>`).

.. autosummary::
   :toctree: generated
   :template: function.rst
   :nosignatures:

   tune_blinks_mahot
   transplant_error_mahot

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   BlinkTuneResult

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
   pupil_response_coverage
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
