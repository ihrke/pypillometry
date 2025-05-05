"""Python-package to help with processing of pupillometric and eyetracking data.

This package provides classes for handling pupillometric and eyetracking data
as well as tools for plotting and analyzing these data.

- Github: https://github.com/ihrke/pypillometry
- Documentation: https://ihrke.github.io/pypillometry

"""

__all__ = ["logging_set_level", "logging_disable", "logging_enable",
           "EyeDataDict", "CachedEyeDataDict", "EyeData", "GazeData", "PupilData", "GenericEyeData",
           "example_datasets", "get_example_data", "get_interval_stats",
           "ERPD", "load_study_osf", "load_study_local",
           "loglevel", "nologging"]

from .eyedata.eyedatadict import EyeDataDict, CachedEyeDataDict
from .eyedata.generic import GenericEyeData
from .eyedata.eyedata import EyeData
from .eyedata.gazedata import GazeData
from .eyedata.pupildata import PupilData
from .convenience import *
from .example_data import example_datasets, get_example_data
from .intervals import get_interval_stats
from .erpd import ERPD
from .io import load_study_osf, load_study_local

## logging
from loguru import logger
import sys
from contextlib import contextmanager
from typing import Optional

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
    logger_format = (
        "<green>pp: {time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        #"<cyan>{name}</cyan>:"
        "<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )    
    logger.add(sys.stderr, format=logger_format, level=level)
    logging_enable()


@contextmanager
def loglevel(level: str):
    """
    Temporarily set the logging level for pypillometry submodules.
    The original level is restored when exiting the context.

    Parameters
    ----------
    level : str
        The temporary logging level. Can be one of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".

    Examples
    --------
    >>> with loglevel("DEBUG"):
    ...     # Code here will use DEBUG level logging
    ...     pass
    >>> # Back to original logging level
    """
    original_level = logger._core.min_level
    logging_set_level(level)
    try:
        yield
    finally:
        logging_set_level(original_level)


@contextmanager
def nologging():
    """
    Temporarily disable logging for pypillometry submodules.
    Logging is re-enabled when exiting the context.

    Examples
    --------
    >>> with nologging():
    ...     # Code here will not produce any logs
    ...     pass
    >>> # Logging is re-enabled
    """
    was_enabled = logger._core.enabled
    logging_disable()
    try:
        yield
    finally:
        if was_enabled:
            logging_enable()


# this enables log-messages with INFO or above
logging_set_level("INFO")

import os.path
__package_path__ = os.path.abspath(os.path.dirname(__file__))
