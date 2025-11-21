"""Python-package to help with processing of pupillometric and eyetracking data.

This package provides classes for handling pupillometric and eyetracking data
as well as tools for plotting and analyzing these data.

Unit Handling:
--------------
Distance and angle parameters accept three formats:
1. Plain numbers (assumed to be mm for distances, radians for angles - with warning)
2. String format: "600 mm", "60 cm", "20 degrees", "-90 degrees"
3. Pint Quantity: 600 * ureg.mm, 20 * ureg.degree

The unit registry is available as `pypillometry.ureg` for advanced users.

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
        '.events',
        '.intervals',
        '.erpd',
        '.io',
        '.logging',
        '.plot',
        '.roi',
        '.signal'
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

# Add imported objects to the global namespace
globals().update(_imported_objects)


## ------------------------------------- 
# --------- units ----------------------
# Import unit registry for advanced users
from .units import ureg
__all__.add('ureg')
__all__ = sorted(list(__all__))

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
