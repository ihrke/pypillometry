"""
eyedata.py
==========

Module containing classes for handling eye data. Several different classes are available for handling
different types of eye data.

- :class:`PupilData`: If you have only pupillometric data, use this class.
- :class:`GazeData`: If you only have gaze data, use this class.
- :class:`EyeData`: If you have both gaze and pupillometric data, use this class.

These classes inherit from :class:`GenericEyeData`, which provides some basic, shared functionality 
for handling eye data.

The data inside each of these classes is stored in the `data` attribute, which is a dictionary-like object
of type :class:`EyeDataDict`. 

Each of these classes provides access to a matching plotter object (from module :mod:`pypillometry.plot`) 
that is stored in the `plot` attribute.

Calibration classes:

- :class:`SpatialCalibration`: Container for eye-tracker spatial calibration data.
- :class:`ForeshorteningCalibration`: Container for foreshortening correction parameters.

Setup classes:

- :class:`ExperimentalSetup`: Geometric model of eye-tracking experimental setup.

Simulation classes:

- :class:`FakeEyeData`: Synthetic eye-tracking data for testing and validation.
"""

__all__ = ["EyeDataDict", "GenericEyeData","GazeData","PupilData", "EyeData", 
           "SpatialCalibration", "ForeshorteningCalibration", "FakeEyeData",
           "ExperimentalSetup"]

from .eyedatadict import EyeDataDict
from .generic import GenericEyeData
from .gazedata import GazeData
from .pupildata import PupilData
from .eyedata import EyeData
from .spatial_calibration import SpatialCalibration
from .foreshortening_calibration import ForeshorteningCalibration
from .fake_eyedata import FakeEyeData
from .experimental_setup import ExperimentalSetup