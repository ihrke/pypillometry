"""
Pypillometry
============

This is a python-package to help with processing of pupillometric data.
"""

__all__ = ["eyedata","plot","signal","convenience","io","erpd"]

from .eyedata.eyedatadict import EyeDataDict



import os.path
__package_path__ = os.path.abspath(os.path.dirname(__file__))
