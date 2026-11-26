"""
:class:`Simulation`: recipe of a simulated recording, built by chaining steps.
"""
import copy

import numpy as np

from ..eyedata.eyedatadict import EyeDataDict
from ..state import Stateful
from ..units import parse_time
from .steps import (STAGE_KEYS, STAGES, Fixations, Foreshortening, SimStep, TimescalePupil,
                    Units, WhiteNoise)
from .task import Task

__all__ = ["Simulation"]


class _Run:
    """Arrays of a simulation while its steps run."""

    def __init__(self, sim, n):
        self.n, self.fs, self.setup = n, sim.fs, sim.setup
        self.eyes = tuple(sim.eyes)
        self.log_size = {e: np.zeros(n) for e in self.eyes}   # latent, log-%
        self.log_pupil = None    # with foreshortening (log-%); set before stage 3
        self.pupil = {}          # in the units of the tracker
        self.x, self.y = {}, {}
        self.events = None       # columns of the events of the task
        self.truth = EyeDataDict()
        self._has_pupil = set()

    def set_pupil(self, eye, values):
        if eye in self._has_pupil:
            raise ValueError(f"more than one pupil step for the {eye} eye")
        self._has_pupil.add(eye)
        self.log_size[eye] = np.asarray(values, dtype=float)
        self.truth[eye, "log_size"] = self.log_size[eye]


class Simulation(Stateful):
    """
    Recipe of a simulated recording.

    A simulation is built by chaining steps; each method returns a new
    :class:`Simulation` with one more step, so that a base simulation can
    be varied::

        base = pp.sim.Simulation("2 min", fs=1000, seed=1).pupil().units(800)
        low, high = base.noise(0.1), base.noise(1.0)

    :meth:`run` makes the data: a :class:`~pypillometry.sim.SimEyeData`
    with the ground truth in ``d.truth`` and the recipe in
    ``d.simulation``. The steps run in a fixed order of stages, whatever
    the order in which they were added: latent pupil size (log-%), gaze,
    foreshortening, units of the tracker, noise (see
    :mod:`pypillometry.sim.steps`). Without a units step, the pupil has a
    median of 1000.

    Parameters
    ----------
    duration : float, str or Quantity, optional
        duration of the recording (plain numbers are ms), e.g. ``"2 min"``;
        default: until the end of the task (see :meth:`task`)
    fs : float
        sampling rate in Hz (default 1000)
    eyes : str or list
        ``"left"`` (default), ``"right"`` or both
    seed : int, optional
        seed of the random numbers; if None, :meth:`run` draws one and
        stores it in the simulation of the data, so that every result can
        be made again
    setup : ExperimentalSetup, optional
        geometry of the setup (needed for :meth:`foreshortening`; the
        screen size is used by :meth:`gaze`)
    name : str, optional
        name of the data (default: a random name)

    Attributes
    ----------
    steps : list
        the steps (see :mod:`pypillometry.sim.steps`)

    Examples
    --------
    >>> import pypillometry as pp
    >>> sim = (pp.sim.Simulation("10 s", fs=100, eyes=["left", "right"], seed=1)
    ...        .pupil(binocular=True).units(800).noise(0.2))
    >>> d = sim.run()
    >>> d.truth.variables[:3]
    ['log_size', 'log_fast', 'log_diff']
    >>> bool((d.simulation.run()["left_pupil"] == d["left_pupil"]).all())
    True
    """
    _STATE_FIELDS = ("duration", "fs", "eyes", "seed", ("setup", "ExperimentalSetup"),
                     ("steps", "simsteps"), "name")

    def __init__(self, duration=None, fs=1000.0, eyes="left", seed=None, setup=None,
                 name=None):
        self.duration = None if duration is None else float(parse_time(duration, to="ms"))
        if self.duration is not None and self.duration <= 0:
            raise ValueError("duration must be positive")
        self.fs = float(fs)
        if self.fs <= 0:
            raise ValueError("fs must be positive")
        self.eyes = [eyes] if isinstance(eyes, str) else list(eyes)
        if not self.eyes or len(set(self.eyes)) != len(self.eyes) \
                or not set(self.eyes) <= {"left", "right"}:
            raise ValueError(f"eyes must be 'left' and/or 'right', got {eyes!r}")
        self.seed = None if seed is None else int(seed)
        self.setup = setup
        self.name = name
        self.steps = []

    def __repr__(self):
        steps = "".join("\n  ." + repr(s).replace("\n", "\n    ") for s in self._ordered())
        seed = "" if self.seed is None else f", seed={self.seed}"
        dur = "duration of the task" if self.duration is None else f"{self.duration / 1000:g} s"
        return f"Simulation({dur} at {self.fs:g} Hz, eyes={self.eyes}{seed}){steps}"

    # -- building ---------------------------------------------------------------
    def add(self, step):
        """
        A copy with one more step.

        Parameters
        ----------
        step : SimStep
            the step (see :mod:`pypillometry.sim.steps`)

        Returns
        -------
        Simulation
        """
        if not isinstance(step, SimStep):
            raise TypeError(f"expected a simulation step, got {type(step).__name__}")
        out = copy.deepcopy(self)
        out.steps.append(step)
        return out

    def pupil(self, *steps, **kwargs):
        """
        Add the latent pupil size: the given steps, or a
        :class:`~pypillometry.sim.steps.TimescalePupil` with the keyword
        arguments (e.g., ``params={"fast_sd": 5.0}``; default: the centre
        of the general prior of the timescale model).

        Returns
        -------
        Simulation
        """
        if steps and kwargs:
            raise ValueError("give either steps or the arguments of TimescalePupil")
        out = self
        for step in steps or [TimescalePupil(**kwargs)]:
            out = out.add(step)
        return out

    def task(self, task, start=None):
        """
        Add the events of a task (see :class:`~pypillometry.sim.Task` and
        :mod:`pypillometry.sim.tasks`).

        The events of the task become the events of the data (``d.events``,
        with their metadata); events outside of the recording are left out.

        Parameters
        ----------
        task : Task
        start : float, str or distribution, optional
            time of the start of the task in the recording (default: that of
            the task, 0)

        Returns
        -------
        Simulation
        """
        if not isinstance(task, Task):
            raise TypeError(f"expected a Task, got {type(task).__name__}")
        if any(s.stage == "events" for s in self.steps):
            raise ValueError("the simulation has a task already")
        return self.add(task if start is None else task._with(start=start))

    def gaze(self, step=None, **kwargs):
        """
        Add the gaze: `step`, or :class:`~pypillometry.sim.steps.Fixations`
        with the keyword arguments.

        Returns
        -------
        Simulation
        """
        return self.add(step if step is not None else Fixations(**kwargs))

    def foreshortening(self, **kwargs):
        """
        Add the foreshortening of the pupil
        (:class:`~pypillometry.sim.steps.Foreshortening`); needs the gaze
        and the `setup` of the simulation.

        Returns
        -------
        Simulation
        """
        return self.add(Foreshortening(**kwargs))

    def units(self, median=1000.0):
        """
        Add the units of the tracker (:class:`~pypillometry.sim.steps.Units`).

        Parameters
        ----------
        median : float or dict
            median pupil size (per eye with a dict)

        Returns
        -------
        Simulation
        """
        return self.add(Units(median))

    def noise(self, sd=0.2, scale="log%", eyes=None):
        """
        Add white measurement noise (:class:`~pypillometry.sim.steps.WhiteNoise`).

        Parameters
        ----------
        sd : float
            standard deviation (log-% by default)
        scale : {"log%", "data"}
            units of `sd`
        eyes : str or list, optional
            eyes (default: all)

        Returns
        -------
        Simulation
        """
        return self.add(WhiteNoise(sd, scale, eyes))

    # -- running ----------------------------------------------------------------
    def _ordered(self):
        steps = list(self.steps)
        if not any(s.stage == "units" for s in steps):
            steps.append(Units())
        return sorted(steps, key=lambda s: STAGES.index(s.stage))

    def run(self, seed=None):
        """
        Simulate the recording.

        Parameters
        ----------
        seed : int, optional
            seed of the random numbers; default: the seed of the
            simulation (or a new one if it has none)

        Returns
        -------
        SimEyeData
            the data, with the ground truth (``d.truth``) and the
            simulation with the seed that was used (``d.simulation``)
        """
        from .simeyedata import SimEyeData
        sim = copy.deepcopy(self)
        if seed is not None:
            sim.seed = int(seed)
        if sim.seed is None:
            sim.seed = int(np.random.default_rng().integers(2**62))
        def rng_of(step, k):
            ss = np.random.SeedSequence(sim.seed, spawn_key=(STAGE_KEYS[step.stage], k))
            return np.random.default_rng(ss)

        ordered = sim._ordered()
        events, end = None, None
        for step in ordered:
            if step.stage == "events":
                events, end = step.generate(rng_of(step, 0))
        duration = sim.duration
        if duration is None:
            if end is None:
                raise ValueError("the simulation needs a duration (or a task)")
            duration = end
        n = int(round(duration * sim.fs / 1000.0))
        if n < 2:
            raise ValueError(f"the recording is too short ({duration:g} ms)")
        run = _Run(sim, n)
        run.events = events
        count = {}
        for step in ordered:
            if step.stage == "events":
                continue
            if run.log_pupil is None and STAGES.index(step.stage) >= STAGES.index("foreshortening"):
                # the measured pupil starts from the latent one (after the
                # pupil and gaze stages)
                run.log_pupil = {e: v.copy() for e, v in run.log_size.items()}
            k = count.get(step.stage, 0)
            count[step.stage] = k + 1
            step.apply(run, rng_of(step, k))
        channels = {}
        for e in sim.eyes:
            channels[f"{e}_pupil"] = run.pupil[e]
            if e in run.x:
                channels[f"{e}_x"], channels[f"{e}_y"] = run.x[e], run.y[e]
        kwargs = {}
        if run.events is not None:
            from ..events import Events
            ev = run.events
            keep = (ev["onset"] >= 0) & (ev["onset"] <= (n - 1) * 1000.0 / sim.fs)
            kwargs["events"] = Events(ev["onset"][keep], ev["label"][keep], units="ms",
                                      meta={k: v[keep] for k, v in ev.items()
                                            if k not in ("onset", "label")})
        return SimEyeData(sampling_rate=sim.fs, **channels, truth=run.truth,
                          simulation=sim, experimental_setup=copy.deepcopy(sim.setup),
                          name=sim.name, **kwargs)
