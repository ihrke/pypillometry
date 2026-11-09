"""
eyedata.py
==========

Module containing the class for handling eye data.

- :class:`EyeData`: pupil and/or gaze data of one or two eyes (any set of
  channels, e.g. only pupil data).

The methods for pupil and gaze data are defined in the modules `pupil` and
`gaze` (the namespaces ``d.pupil`` and ``d.gaze``); the decorator that
records their calls in the history is in `history`.

The data are stored in the `data` attribute, which is a dictionary-like object
of type :class:`EyeDataDict`. 

The plotting methods (module :mod:`pypillometry.plot`) are available in the
`plot` attribute.

Calibration classes:

- :class:`SpatialCalibration`: Container for eye-tracker spatial calibration data.
- :class:`ForeshorteningCalibration`: Container for foreshortening correction parameters.

Setup classes:

- :class:`ExperimentalSetup`: Geometric model of eye-tracking experimental setup.

Simulation classes:

- :class:`FakeEyeData`: Synthetic eye-tracking data for testing and validation.
"""

__all__ = ["EyeDataDict", "MaskPolicy", "EyeData",
           "SpatialCalibration", "ForeshorteningCalibration", "FakeEyeData",
           "ExperimentalSetup", "keephistory"]

from .eyedatadict import EyeDataDict, MaskPolicy
from .history import keephistory
from .eyedata import EyeData
from .spatial_calibration import SpatialCalibration
from .foreshortening_calibration import ForeshorteningCalibration
from .fake_eyedata import FakeEyeData
from .experimental_setup import ExperimentalSetup