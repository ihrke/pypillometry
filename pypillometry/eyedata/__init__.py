"""
eyedata.py
==========

Module containing classes for handling eye data.

- :class:`pypillometry.eyedata.EyeDataDict`: A dictionary-like object for storing eye data.
- :class:`pypillometry.eyedata.GenericEyeData`: A generic class for handling eye data.
- :class:`pypillometry.eyedata.GazeData`: A class for handling gaze data.
"""

__all__ = ["EyeDataDict", "GenericEyeData","GazeData","PupilData", "EyeData"]

from .eyedatadict import EyeDataDict
from .generic import GenericEyeData
from .gazedata import GazeData
from .pupildata import PupilData
from .eyedata import EyeData