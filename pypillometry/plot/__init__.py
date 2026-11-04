"""
Plotting namespace of :class:`~pypillometry.EyeData` objects. This module does
not have to be imported directly: the plots are called through ``d.plot``,
e.g. ``d.plot.plot_timeseries()``.

The plots are written in one module per domain (:mod:`.generic`,
:mod:`.pupil`, :mod:`.gaze`); :class:`PlotNamespace` combines them.
"""
from .pupil import PupilPlots
from .gaze import GazePlots


class PlotNamespace(PupilPlots, GazePlots):
    """
    Plotting methods of an :class:`~pypillometry.EyeData` object, used as
    ``d.plot`` (e.g., ``d.plot.plot_timeseries()``).

    Like ``d.pupil`` and ``d.gaze``, it is a view on the data object with no
    state of its own (see :class:`~pypillometry.eyedata.namespace.Namespace`).
    It combines the generic plots, the pupil plots and the gaze plots, which
    are written in separate modules. Plots that need channels the object does
    not have raise a ValueError.
    """
    _name = "plot"
