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

# Each submodule defines its public API in its own ``__all__``. The star
# imports below are static, so IDEs and type checkers can follow them.
from . import (
    config, units, logging, convenience, eyedata, plot, signal, events,
    intervals, segmented, io, study, roi, example_data, viewer,
)
from .config import *
from .units import *
from .logging import *
from .convenience import *
from .eyedata import *
from .plot import *
from .signal import *
from .events import *
from .intervals import *
from .segmented import *
from .io import *
from .study import *
from .roi import *
from .example_data import *
from .viewer import *

__all__ = []
__all__ += config.__all__
__all__ += units.__all__
__all__ += logging.__all__
__all__ += convenience.__all__
__all__ += eyedata.__all__
__all__ += plot.__all__
__all__ += signal.__all__
__all__ += events.__all__
__all__ += intervals.__all__
__all__ += segmented.__all__
__all__ += io.__all__
__all__ += study.__all__
__all__ += roi.__all__
__all__ += example_data.__all__
__all__ += viewer.__all__

# loguru logger, available as ``pypillometry.logger`` (not part of __all__)
from loguru import logger

initialize_logging()

__package_path__ = os.path.abspath(os.path.dirname(__file__))
