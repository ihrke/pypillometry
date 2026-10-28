"""Python-package to help with processing of pupillometric and eyetracking data.

This package provides classes for handling pupillometric and eyetracking data
as well as tools for plotting and analyzing these data.

Unit Handling:
--------------
Times, distances and angles can carry units, either from the `pypillometry.u`
namespace (``3 * u.min``, ``(-200, 800) * u.ms``, ``60 * u.cm``,
``20 * u.deg``) or as strings (``"3 min"``, ``"60 cm"``, ``"20 deg"``).

- Github: https://github.com/ihrke/pypillometry
- Documentation: https://ihrke.github.io/pypillometry

"""

import os.path

# ---------------------------------------------------------------------------
# Data classes
from .eyedata import (
    GenericEyeData, EyeData, PupilData, GazeData, FakeEyeData,
    EyeDataDict, ExperimentalSetup, SpatialCalibration,
    ForeshorteningCalibration, keephistory,
)
from .plot import EyePlotter, PupilPlotter, GazePlotter
from .segmented import (
    SegmentedEyeData, GroupLevelSegmentedData, group_segments,
)
from .events import Events
from .intervals import Intervals, merge_intervals
from .roi import ROI, CircularROI, RectangularROI

# ---------------------------------------------------------------------------
# Reading and writing data
from .io import (
    read_eyelink, read_pickle, write_pickle, download, fetch_url, local_copy,
    osf_authenticate,
)
from .study import load_study_local, load_study_osf, get_osf_project_files
from .example_data import (
    get_available_datasets, get_example_data, get_rlmw_002,
    get_rlmw_002_short, get_rlmw_010_edf,
)

# ---------------------------------------------------------------------------
# Signal processing
from .signal import (
    # baseline
    bspline, butter_lowpass, butter_lowpass_filter, downsample,
    baseline_envelope, baseline_envelope_iter_bspline, baseline_pupil_model,
    # preprocessing
    smooth_window, velocity_savgol, detect_blinks_velocity,
    detect_blinks_zero, blink_onsets_mahot,
    # pupil response
    lowpass_filter_iterative, pupil_signal_quality, pupil_kernel,
    pupil_kernel_t, pupil_get_max_duration, plot_prf,
    pupil_build_design_matrix, pupil_response, pupilresponse_nnls,
    pupil_eye_offset,
    # synthetic data
    fake_pupil_baseline, fake_gaze_fixations, add_measurement_noise,
    generate_foreshortening_data,
)
from .convenience import zscale

# ---------------------------------------------------------------------------
# Viewer
from .viewer import view, tweak

# ---------------------------------------------------------------------------
# Units, settings and logging
from .units import u, Unit, Quantity, parse_time, parse_distance, parse_angle
from .config import set_default_inplace, get_default_inplace
from .logging import (
    logging_disable, logging_enable, logging_set_level, logging_get_level,
    loglevel, nologging, initialize_logging,
)
from loguru import logger  # available as pp.logger, not part of __all__

__all__ = [
    # data classes
    "GenericEyeData", "EyeData", "PupilData", "GazeData", "FakeEyeData",
    "EyeDataDict", "ExperimentalSetup", "SpatialCalibration",
    "ForeshorteningCalibration", "keephistory",
    "EyePlotter", "PupilPlotter", "GazePlotter",
    "SegmentedEyeData", "GroupLevelSegmentedData", "group_segments",
    "Events",
    "Intervals", "merge_intervals",
    "ROI", "CircularROI", "RectangularROI",
    # reading and writing data
    "read_eyelink", "read_pickle", "write_pickle", "download", "fetch_url",
    "local_copy", "osf_authenticate",
    "load_study_local", "load_study_osf", "get_osf_project_files",
    "get_available_datasets", "get_example_data", "get_rlmw_002",
    "get_rlmw_002_short", "get_rlmw_010_edf",
    # signal processing
    "bspline", "butter_lowpass", "butter_lowpass_filter", "downsample",
    "baseline_envelope", "baseline_envelope_iter_bspline",
    "baseline_pupil_model",
    "smooth_window", "velocity_savgol", "detect_blinks_velocity",
    "detect_blinks_zero", "blink_onsets_mahot",
    "lowpass_filter_iterative", "pupil_signal_quality", "pupil_kernel",
    "pupil_kernel_t", "pupil_get_max_duration", "plot_prf",
    "pupil_build_design_matrix", "pupil_response", "pupilresponse_nnls",
    "pupil_eye_offset",
    "fake_pupil_baseline", "fake_gaze_fixations", "add_measurement_noise",
    "generate_foreshortening_data",
    "zscale",
    # viewer
    "view", "tweak",
    # units, settings and logging
    "u", "Unit", "Quantity", "parse_time", "parse_distance", "parse_angle",
    "set_default_inplace", "get_default_inplace",
    "logging_disable", "logging_enable", "logging_set_level",
    "logging_get_level", "loglevel", "nologging", "initialize_logging",
]

initialize_logging()

__package_path__ = os.path.abspath(os.path.dirname(__file__))
