"""pypillometry - processing of pupillometric and eyetracking data.

Reads recordings from eye trackers, cleans them (blinks, artifacts,
filtering), cuts them into event-related segments and plots the results.
Pupil size and gaze of one or both eyes are kept together in one object
that records its own processing history.

Example
-------
::

    import pypillometry as pp
    from pypillometry import u

    d = pp.get_example_data("rlmw_002")
    d = d | pp.pupil.blinks_detect() | pp.pupil.blinks_interpolate()
    epochs = d.get_epochs("S_", (-500, 2500) * u.ms)

Overview
--------
EyeData
    The main data class; ``SegmentedEyeData`` for data cut into segments.
pupil, gaze
    Pipeline steps for pupil and gaze data, used as ``d | pp.pupil.<step>()``.
    General steps (``crop``, ``downsample``, ...) are at the top level.
read_eyelink, read_file, load_study_local, get_example_data
    Reading and writing data.
Events, Intervals, Epochs
    Event markers, time intervals and event-related segments.
signal
    Signal-processing functions that work on plain numpy arrays.
u
    Units for times, distances and angles (``3 * u.min``, ``60 * u.cm``);
    strings like ``"200 ms"`` work too. See `pypillometry.units`.
view
    Interactive viewer for long recordings.

- Github: https://github.com/ihrke/pypillometry
- Documentation: https://ihrke.github.io/pypillometry
"""

import os.path

# ---------------------------------------------------------------------------
# Data classes
from .eyedata import (
    EyeData, FakeEyeData,
    EyeDataDict, MaskPolicy, ExperimentalSetup, SpatialCalibration,
    ForeshorteningCalibration, keephistory,
)
from .eyedata.pipes import Step, Pipeline

# ---------------------------------------------------------------------------
# Pipeline steps: d | pp.pupil.blinks_detect() | pp.crop(...) | ...
from .eyedata.steps import (
    pupil, gaze,
    crop, downsample, scale, unscale, merge_eyes, reset_time,
    mask_intervals, interpolate_intervals, exclude_intervals, keep_intervals,
    remove_intervals, merge_masks, clear_layer, set_blinks, set_events,
    add_events, set_experimental_setup, drop_data, drop_original,
    fill_time_discontinuities,
)
from .segmented import (
    SegmentedEyeData, concat_segments, group_segments,
)
from .events import Events, ClockSync, sync_clocks
from .intervals import Intervals, concat_intervals
from .epochs import Epochs
from .table import Table, RecordDict
from .roi import ROI, CircularROI, RectangularROI
from .quality import QualityReport, QualityMetric

# ---------------------------------------------------------------------------
# Reading and writing data
from .io import (
    read_eyelink, read_file, write_file, download, fetch_url, local_copy,
    osf_authenticate,
)
from .study import load_study_local, load_study_osf, get_osf_project_files
from .example_data import (
    get_available_datasets, get_example_data, get_rlmw_002,
    get_rlmw_002_short, get_rlmw_010_edf,
)

# ---------------------------------------------------------------------------
# Signal processing for numpy arrays: pp.signal.<function>
from . import signal

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
    "EyeData", "FakeEyeData",
    "EyeDataDict", "MaskPolicy", "ExperimentalSetup", "SpatialCalibration",
    "ForeshorteningCalibration", "keephistory", "Step", "Pipeline",
    # pipeline steps
    "pupil", "gaze",
    "crop", "downsample", "scale", "unscale", "merge_eyes", "reset_time",
    "mask_intervals", "interpolate_intervals", "exclude_intervals",
    "keep_intervals", "remove_intervals", "merge_masks", "clear_layer",
    "set_blinks", "set_events", "add_events", "set_experimental_setup",
    "drop_data", "drop_original", "fill_time_discontinuities",
    "SegmentedEyeData", "concat_segments", "group_segments",
    "Events", "ClockSync", "sync_clocks",
    "Intervals", "concat_intervals", "Epochs", "Table", "RecordDict",
    "ROI", "CircularROI", "RectangularROI",
    "QualityReport", "QualityMetric",
    # reading and writing data
    "read_eyelink", "read_file", "write_file", "download", "fetch_url",
    "local_copy", "osf_authenticate",
    "load_study_local", "load_study_osf", "get_osf_project_files",
    "get_available_datasets", "get_example_data", "get_rlmw_002",
    "get_rlmw_002_short", "get_rlmw_010_edf",
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
