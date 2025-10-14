"""
logging.py
==========

Logging functionality for pypillometry package.

This module provides comprehensive logging control for the pypillometry package,
including level management, context managers, and integration with external
libraries like cmdstanpy.
"""

from loguru import logger
import sys
import logging
from contextlib import contextmanager
from typing import Optional

# Configure cmdstanpy logging (disabled by default)
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True


def logging_disable():
    """
    Disable logging for all pypillometry submodules.
    """
    logger.disable("pypillometry")
    cmdstanpy_logger.disabled = True


def logging_enable():
    """
    Enable logging for all pypillometry submodules.
    """
    logger.enable("pypillometry")


def logging_set_level(level="INFO"):
    """
    Set the logging level for all pypillometry submodules.

    In the case of DEBUG, the logging for cmdstanpy is also enabled.

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
    
    # Enable cmdstanpy logging during DEBUG level
    if level == "DEBUG":
        cmdstanpy_logger.disabled = False
    else:
        cmdstanpy_logger.disabled = True


def logging_get_level():
    """
    Get the current logging level for pypillometry.
    
    Returns
    -------
    str
        The current logging level as a string (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    """
    try:
        # Get the numeric level from loguru
        numeric_level = logger._core.min_level
        
        # Use loguru's level system to get the name
        for level_name in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            if logger.level(level_name).no == numeric_level:
                return level_name
        
        # Fallback for custom levels
        return f"LEVEL_{numeric_level}"
    except Exception:
        return "UNKNOWN"


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
    original_level = logging_get_level()  # This returns a string, not a float
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


def initialize_logging():
    """
    Initialize logging with default settings.
    
    This function is called automatically when the package is imported.
    """
    logging_set_level("INFO")
