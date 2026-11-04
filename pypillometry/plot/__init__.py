"""
Module containing classes for plotting eye data. This module does not have to
be imported directly. Rather, each :class:`~pypillometry.EyeData` object has a
`plot` attribute that contains a plotter object
(:class:`~pypillometry.EyePlotter`, with the methods of
:class:`~pypillometry.PupilPlotter` and :class:`~pypillometry.GazePlotter`),
so that plotting methods are called as, e.g., ``d.plot.plot_timeseries()``.
"""
__all__=["GazePlotter","EyePlotter","PupilPlotter"]

from .gaze import GazePlotter
from .eyeplotter import EyePlotter
from .pupil import PupilPlotter