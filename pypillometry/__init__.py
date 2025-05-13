"""Python-package to help with processing of pupillometric and eyetracking data.

This package provides classes for handling pupillometric and eyetracking data
as well as tools for plotting and analyzing these data.

- Github: https://github.com/ihrke/pypillometry
- Documentation: https://ihrke.github.io/pypillometry

"""

import os.path
import importlib
import inspect
from typing import List, Set, Dict, Any

def _collect_and_import_submodules() -> Dict[str, Any]:
    """Collect all public names and import everything from submodules."""
    public_names = set()
    imported_objects = {}
    
    # List of submodules to import from
    submodules = [
        '.eyedata.eyedatadict',
        '.eyedata.generic',
        '.eyedata.eyedata',
        '.eyedata.gazedata',
        '.eyedata.pupildata',
        '.convenience',
        '.example_data',
        '.intervals',
        '.erpd',
        '.io',
        '.plot'
    ]
    
    for module_name in submodules:
        try:
            # Import the module
            module = importlib.import_module(module_name, package='pypillometry')
            
            # Get all public names (not starting with _)
            names = [name for name in dir(module) if not name.startswith('_')]
            public_names.update(names)
            
            # Import all public objects
            for name in names:
                try:
                    imported_objects[name] = getattr(module, name)
                except AttributeError:
                    print(f"Warning: Could not import {name} from {module_name}")
                    
        except ImportError as e:
            print(f"Warning: Could not import {module_name}: {e}")
    
    return public_names, imported_objects

# Collect all public names and import everything
__all__, _imported_objects = _collect_and_import_submodules()
__all__ = sorted(list(__all__))

# Add imported objects to the global namespace
globals().update(_imported_objects)

## logging
from loguru import logger
import sys
from contextlib import contextmanager
from typing import Optional

## logging for cmdstanpy disabled by default
import logging
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

__package_path__ = os.path.abspath(os.path.dirname(__file__))
