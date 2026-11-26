"""
State-space models of the slow and fast fluctuations of the pupil size
(experimental).

- :class:`TimescaleModel`: model of one recording (one or both eyes) with a
  fast component and slow components at fixed time scales; fit, checks,
  plots, the slow part of the signal and reconstruction of missing data at
  the original samples
- :class:`TimescaleGroup`: group-level fit of several recordings with
  shrinkage of the individual estimates

The mathematics is in :mod:`pypillometry.signal.gaussmarkov` (modules
``bank``, ``hierarchical`` and ``check``), which works on plain arrays.
"""
from .model import TimescaleModel
from .group import TimescaleGroup
from .plots import TimescalePlots, TimescaleGroupPlots

__all__ = ["TimescaleModel", "TimescaleGroup"]
