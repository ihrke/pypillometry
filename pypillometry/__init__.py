"""
Pypillometry
============

This is a python-package to help with processing of pupillometric data.
"""

__all__ = ["eyedata","plot","signal","convenience","io","erpd", "parameters", 
           "logging_set_level", "logging_disable", "logging_enable"]

from .eyedata.eyedatadict import EyeDataDict
from .eyedata.eyedata import EyeData
from .eyedata.gazedata import GazeData
from .eyedata.pupildata import PupilData
from .convenience import *
from .parameters import Parameters

## logging
from loguru import logger
import sys

def logging_disable():
    """
    Disable logging for all pypillometry submodules.
    """
    logger.disable("pypillometry")

def logging_enable():
    """
    Enable logging for all pypillometry submodules.
    """
    logger.enable("pypillometry")


def logging_set_level(level="INFO"):
    """
    Set the logging level for all pypillometry submodules.

    Parameters
    ----------
    level : str
        The logging level. Can be one of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
    """
    logger.remove()
    logger.add(sys.stderr, level=level)
    logging_enable()


# this enables log-messages with INFO or above
logging_set_level("INFO")

import os.path
__package_path__ = os.path.abspath(os.path.dirname(__file__))
