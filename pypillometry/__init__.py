"""
Pypillometry
============

This is a python-package to help with processing of pupillometric data.
"""

__all__ = ["eyedata","plot","signal","convenience","io","erpd"]

from .eyedata.eyedatadict import EyeDataDict
from .eyedata.eyedata import EyeData
from .eyedata.gazedata import GazeData
from .eyedata.pupildata import PupilData
from .convenience import *

import os.path
__package_path__ = os.path.abspath(os.path.dirname(__file__))
