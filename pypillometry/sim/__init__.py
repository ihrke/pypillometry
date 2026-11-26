"""
Simulated eye-tracking data with ground truth.

A :class:`Simulation` is built by chaining steps and run to get a
:class:`SimEyeData` object, which holds the data together with the true
time series (``d.truth``) and the recipe that made them
(``d.simulation``)::

    import pypillometry as pp
    sim = (pp.sim.Simulation("2 min", fs=500, eyes=["left", "right"], seed=1)
           .pupil(binocular=True)       # latent pupil (timescale model)
           .units(median=800)           # units of the tracker
           .noise(0.2))                 # white noise (log-%)
    d = sim.run()

- :class:`Simulation`: the recipe (duration, sampling rate, eyes, seed,
  experimental setup and steps)
- :class:`Task`: the design of an experiment (blocks of trials with stimuli
  and responses), which makes the events; ready-made tasks in
  :mod:`~pypillometry.sim.tasks` (PVT, SART, FT-RSGT, n-back)
- :mod:`~pypillometry.sim.steps`: the steps (:class:`TimescalePupil`,
  :class:`Fixations`, :class:`Foreshortening`, :class:`Units`,
  :class:`WhiteNoise`)
- :class:`SimEyeData`: the simulated data
"""
from .steps import TimescalePupil, Fixations, Foreshortening, Units, WhiteNoise
from .dist import parse_dist
from .task import Task
from .simulation import Simulation
from .simeyedata import SimEyeData, SimPlotNamespace
from . import tasks

__all__ = ["Simulation", "SimEyeData", "SimPlotNamespace", "Task", "parse_dist",
           "TimescalePupil", "Fixations", "Foreshortening", "Units", "WhiteNoise"]
