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
from .parameters import Parameters

## logging
from loguru import logger
# this disables the logger by default, the user can enable it by calling `logger.enable("pypillometry")`
logger.disable("pypillometry")


import os.path
__package_path__ = os.path.abspath(os.path.dirname(__file__))
