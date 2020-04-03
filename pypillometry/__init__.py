"""
Pypillometry
============

This is a python-package to help with processing of pupillometric data.
"""
from .baseline import *
from .convenience import *
from .fakedata import *
from .pupil import *
from .pupildata import *
from .erpd import *

import os.path
__package_path__ = os.path.abspath(os.path.dirname(__file__))

