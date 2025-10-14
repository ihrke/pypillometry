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
        '.logging',
        '.plot',
        '.roi'
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


## ------------------------------------- 
# --------- logging --------------------
# Import logging functionality from submodule
from .logging import (
    logging_disable, logging_enable, logging_set_level, logging_get_level,
    loglevel, nologging, initialize_logging
)

# Import logger for use throughout the package
from loguru import logger

# Initialize logging with default settings
initialize_logging()

__package_path__ = os.path.abspath(os.path.dirname(__file__))
