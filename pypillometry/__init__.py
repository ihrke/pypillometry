"""
Pypillometry
============

This is a python-package to help with processing of pupillometric data.
"""

_inplace=False ## default for whether or not inplace-operations should be used

from .baseline import *
from .convenience import *
from .fakedata import *
from .pupil import *
from .eyedata_generic import *
from .pupildata import *
from .eyedata import *
from .erpd import *

import os.path
__package_path__ = os.path.abspath(os.path.dirname(__file__))
