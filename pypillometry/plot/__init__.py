"""
plot.py
=======

Module containing classes for plotting eye data. This module does not have to
be imported directly. Rather, each of the classes from `eyedata` has a `plot`
attribute that contains a plotter object for that class.
"""
__all__=["GazePlotter","EyePlotter","PupilPlotter"]

from .gazeplotter import GazePlotter
from .eyeplotter import EyePlotter
from .pupilplotter import PupilPlotter