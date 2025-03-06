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
"""

__all__ = ["EyeDataDict", "GenericEyeData","GazeData","PupilData", "EyeData"]

from .eyedatadict import EyeDataDict
from .generic import GenericEyeData
from .gazedata import GazeData
from .pupildata import PupilData
from .eyedata import EyeData