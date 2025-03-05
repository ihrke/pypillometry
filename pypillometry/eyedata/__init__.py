"""
eyedata.py
==========

Module containing classes for handling eye data. The following classes are available for handling
different types of eye data:

- :class:`EyeDataDict`: A dictionary-like object for storing eye data.
- :class:`GenericEyeData`: A generic class for handling eye data.
- :class:`GazeData`: A class for handling gaze data.
"""

__all__ = ["EyeDataDict", "GenericEyeData","GazeData","PupilData", "EyeData"]

from .eyedatadict import EyeDataDict
from .generic import GenericEyeData
from .gazedata import GazeData
from .pupildata import PupilData
from .eyedata import EyeData