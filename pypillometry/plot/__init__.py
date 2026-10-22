"""
Module containing classes for plotting eye data. This module does not have to
be imported directly. Rather, each data class has a `plot` attribute that
contains a matching plotter object (:class:`~pypillometry.GazeData`:
:class:`~pypillometry.GazePlotter`, :class:`~pypillometry.PupilData`:
:class:`~pypillometry.PupilPlotter`, :class:`~pypillometry.EyeData`:
:class:`~pypillometry.EyePlotter`), so that plotting methods are called
as, e.g., ``d.plot.plot_timeseries()``.
"""
__all__=["GazePlotter","EyePlotter","PupilPlotter"]

from .gazeplotter import GazePlotter
from .eyeplotter import EyePlotter
from .pupilplotter import PupilPlotter