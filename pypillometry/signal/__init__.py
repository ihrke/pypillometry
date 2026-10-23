"""
Stateless, low-level signal-processing functions operating on numpy arrays.

These functions are used by the data classes (e.g.,
:class:`~pypillometry.PupilData`) and generally work in samples or
milliseconds; see the individual docstrings for units.

- :mod:`~pypillometry.signal.baseline`: filtering, downsampling and
  baseline estimation
- :mod:`~pypillometry.signal.preproc`: smoothing, velocity and blink detection
- :mod:`~pypillometry.signal.pupil`: pupil-response function (PRF),
  response estimation and signal quality
- :mod:`~pypillometry.signal.fake`: generation of synthetic data
"""
__all__ = [
    # from baseline.py
    "butter_lowpass_filter", "downsample", "baseline_envelope_iter_bspline",
    # from preproc.py
    "smooth_window", "velocity_savgol", "detect_blinks_velocity", "detect_blinks_zero",
    # from pupil.py
    "pupil_eye_offset",
    # from fake.py
    "fake_pupil_baseline", "fake_gaze_fixations", "add_measurement_noise", 
    "generate_foreshortening_data",
    ]

from .baseline import *
from .preproc import *
from .pupil import *
from .fake import *