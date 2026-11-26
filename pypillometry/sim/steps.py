"""
Steps of a :class:`~pypillometry.sim.Simulation`.

Each step is a small dataclass with the parameters of one part of the
simulated recording. A simulation runs its steps in a fixed order of
stages, whatever the order in which they were added:

1. ``"events"``: the events of a task (:class:`~pypillometry.sim.Task`)
2. ``"pupil"``: the latent pupil size in log-%, :math:`100 \\log(\\text{size}
   / \\text{median})` (:class:`TimescalePupil`)
3. ``"gaze"``: the gaze position in pixels (:class:`Fixations`)
4. ``"foreshortening"``: the apparent pupil size as seen by the camera
   (:class:`Foreshortening`)
5. ``"units"``: from log-% to the units of the tracker (:class:`Units`)
6. ``"occlusion"``: blinks, the lid covers the pupil (:class:`Blinks`)
7. ``"noise"``: measurement noise (:class:`WhiteNoise`)
8. ``"missing"``: losses of the pupil by the tracker (:class:`TrackerLoss`);
   then the samples without pupil (in blinks and losses) are set missing,
   in the pupil and the gaze

Each step draws from its own random stream (derived from the seed of the
simulation, its stage and its position in the stage), so that changing one
step leaves the random numbers of the others unchanged.
"""
from dataclasses import dataclass, fields

import numpy as np

from ..units import parse_time

__all__ = ["TimescalePupil", "Fixations", "Foreshortening", "Units", "WhiteNoise", "Blinks",
           "TrackerLoss"]

#: order in which the stages are run
STAGES = ("events", "pupil", "gaze", "foreshortening", "units", "occlusion", "noise",
          "missing")
#: key of the random stream of each stage (fixed, so that adding stages does
#: not change the random numbers of the others)
STAGE_KEYS = {"pupil": 0, "gaze": 1, "foreshortening": 2, "units": 3, "noise": 4,
              "events": 5, "occlusion": 6, "missing": 7}


def _eyes(eyes):
    """tuple of eyes (None: all eyes of the simulation)"""
    if eyes is None:
        return None
    return (eyes,) if isinstance(eyes, str) else tuple(eyes)


def _plain(value):
    """value with tuples as lists and numpy scalars as numbers (for the state)"""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class SimStep:
    """
    Base class of the steps.

    A step has a `stage` (see :data:`STAGES`), the eyes it applies to
    (`eyes`; None: all eyes of the simulation) and a method ``apply(run,
    rng)`` that changes the arrays of the running simulation and adds to
    its ground truth.
    """
    stage = None

    def __post_init__(self):
        # lists (e.g., from a saved state) as tuples, so that steps compare equal
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, list):
                setattr(self, f.name, tuple(v))
            elif f.name == "eyes" and isinstance(v, str):
                setattr(self, f.name, (v,))

    def eyes_in(self, run):
        """the eyes of the simulation the step applies to"""
        eyes = _eyes(getattr(self, "eyes", None))
        if eyes is None:
            return run.eyes
        unknown = set(eyes) - set(run.eyes)
        if unknown:
            raise ValueError(f"{type(self).__name__}: the simulation has no eyes "
                             f"{sorted(unknown)} (eyes: {list(run.eyes)})")
        return eyes

    def to_dict(self):
        """name and parameters of the step (numbers, strings, lists, dicts)"""
        return {"step": type(self).__name__,
                "params": {f.name: _plain(getattr(self, f.name)) for f in fields(self)}}

    @staticmethod
    def from_dict(d):
        """the step from :meth:`to_dict`"""
        from .task import Task
        cls = {c.__name__: c for c in (TimescalePupil, Fixations, Foreshortening, Units,
                                       WhiteNoise, Task, Blinks, TrackerLoss)}.get(d["step"])
        if cls is None:
            raise ValueError(f"unknown simulation step {d['step']!r}")
        return cls(**d["params"])

    def __repr__(self):
        def short(v):
            if isinstance(v, float):
                return str(int(v)) if v.is_integer() and abs(v) < 1e12 else f"{v:.4g}"
            if isinstance(v, dict):
                return "{" + ", ".join(f"{k!r}: {short(x)}" for k, x in v.items()) + "}"
            if isinstance(v, tuple):
                return "(" + ", ".join(short(x) for x in v) + ("," if len(v) == 1 else "") + ")"
            return repr(v)
        args = ", ".join(f"{f.name}={short(getattr(self, f.name))}" for f in fields(self)
                         if getattr(self, f.name) != f.default)
        return f"{type(self).__name__}({args})"


@dataclass(repr=False)
class TimescalePupil(SimStep):
    """
    Latent pupil size from the timescale model (see
    :class:`~pypillometry.TimescaleModel`): a fast component (Matérn-3/2)
    and slow components ("atoms") at fixed time scales, in log-%.

    Parameters
    ----------
    params : dict, optional
        parameters on their natural scale, as in
        :attr:`TimescaleModel.params <pypillometry.TimescaleModel.params>`:
        ``fast_sd`` (log-%), ``fast_ell`` (ms) and the standard deviation of
        each atom (``sd_3s``, ``sd_10s``, ...; log-%); for two eyes with
        `binocular` also ``diff_sd``, ``diff_ell`` (the difference between
        the eyes) and ``gain`` (of the common components in the second eye
        relative to the first). Missing parameters are the centre of the
        general prior (see
        :func:`~pypillometry.signal.gaussmarkov.general_prior`); the noise
        (``noise_sd``) is ignored (see :class:`WhiteNoise`).
    timescales : sequence of float, str or Quantity
        time scales of the atoms (plain numbers are ms); default 3, 10, 30,
        100 and 300 s
    nu : float
        smoothness of the atoms (default 1.5)
    binocular : bool
        if True, the eyes share the fast component and the atoms and differ
        by a difference component (needs two eyes); if False (default),
        each eye is drawn independently with the same parameters
    cutoff : float, str or Quantity
        time scale from which on the atoms count as slow, for the ground
        truth ``log_slow`` (plain numbers are ms; default 10 s)
    eyes : str or list, optional
        eyes the step applies to (default: all eyes of the simulation)

    Notes
    -----
    Ground truth per eye (log-%): ``log_size`` (the latent signal),
    ``log_fast``, one entry per atom (``log_3s``, ...), ``log_slow`` (the
    atoms from `cutoff`) and, if `binocular`, ``log_diff``; ``log_size``
    is the sum of the components.
    """
    params: dict = None
    timescales: tuple = ("3 s", "10 s", "30 s", "100 s", "300 s")
    nu: float = 1.5
    binocular: bool = False
    cutoff: str = "10 s"
    eyes: tuple = None
    stage = "pupil"

    def bank(self, eyes):
        from ..signal.gaussmarkov import TimescaleBank
        ts = [float(parse_time(t, to="ms")) for t in self.timescales]
        return TimescaleBank(ts, nu=self.nu,
                             channels=tuple(eyes) if self.binocular else 1)

    def theta(self, bank):
        """parameter vector of `bank` (the noise is negligible)"""
        from ..signal.gaussmarkov import general_prior
        p = bank.params(general_prior(bank).mean)
        given = dict(self.params or {})
        unknown = set(given) - set(p)
        if unknown:
            raise ValueError(f"TimescalePupil: unknown parameters {sorted(unknown)}; "
                             f"use {sorted(k for k in p if not k.startswith('noise'))}")
        p.update(given)
        for k in p:
            if k.startswith("noise_sd"):
                p[k] = 1e-9
        for k in ("fast_sd", "diff_sd"):
            if k in p:
                p[k] = max(float(p[k]), 1e-9)
        return bank.theta(**p)

    def apply(self, run, rng):
        eyes = self.eyes_in(run)
        if self.binocular and len(eyes) != 2:
            raise ValueError(f"TimescalePupil(binocular=True) needs two eyes, got {list(eyes)}")
        cutoff = float(parse_time(self.cutoff, to="ms"))
        groups = [eyes] if self.binocular else [(e,) for e in eyes]
        rngs = rng.spawn(len(groups))
        for group, r in zip(groups, rngs):
            bank = self.bank(group)
            theta = self.theta(bank)
            _, X = bank.simulate(theta, run.n, run.fs, rng=r, return_states=True)
            names = ["fast"] + (["diff"] if self.binocular else []) + bank.atom_names
            for ch, eye in enumerate(group):
                parts = {n: X @ bank.select(n, cutoff, channel=ch, theta=theta)
                         for n in names + ["slow", "signal"]}
                run.set_pupil(eye, parts.pop("signal"))
                for n, v in parts.items():
                    run.truth[eye, f"log_{n}"] = v


@dataclass(repr=False)
class Fixations(SimStep):
    """
    Gaze as a sequence of fixations at random places on the screen (the
    same for all eyes).

    Parameters
    ----------
    mean, sd : float, str or Quantity
        mean and standard deviation of the duration of the fixations (plain
        numbers are ms; at least 50 ms); default 300 and 100 ms
    screen : tuple of int, optional
        width and height of the screen in pixels; default: the screen of
        the experimental setup of the simulation, or (1920, 1080)
    """
    mean: str = "300 ms"
    sd: str = "100 ms"
    screen: tuple = None
    stage = "gaze"

    def apply(self, run, rng):
        if self.screen is not None:
            width, height = self.screen
        elif run.setup is not None and run.setup.screen_resolution is not None:
            width, height = run.setup.screen_resolution
        else:
            width, height = 1920, 1080
        mean = float(parse_time(self.mean, to="ms"))
        sd = float(parse_time(self.sd, to="ms"))
        total = run.n * 1000.0 / run.fs
        n_fix = int(total / mean) + 1
        durations = np.clip(rng.normal(mean, sd, n_fix), 50, None)
        durations *= total / durations.sum()        # fill the whole recording
        starts = np.r_[0, np.cumsum(durations)[:-1]]
        idx = np.searchsorted(starts, np.arange(run.n) * 1000.0 / run.fs, side="right") - 1
        x = rng.uniform(0, width, n_fix)[idx]
        y = rng.uniform(0, height, n_fix)[idx]
        for eye in run.eyes:
            run.x[eye], run.y[eye] = x.copy(), y.copy()


@dataclass(repr=False)
class Foreshortening(SimStep):
    """
    Pupil foreshortening: the camera sees the pupil at the angle
    :math:`\\alpha` between the directions from the eye to the camera and
    to the gaze position, so that the measured size is :math:`A_0
    \\cos\\alpha`. Needs the gaze (e.g., :class:`Fixations`) and an
    experimental setup with screen, eye distance and camera position.

    Parameters
    ----------
    eye_offset : float
        horizontal offset of the eye from the screen centre (mm)

    Notes
    -----
    Ground truth per eye: ``cosalpha``.
    """
    eye_offset: float = 0.0
    stage = "foreshortening"

    def apply(self, run, rng):
        from ..eyedata.foreshortening_calibration import _compute_cos_alpha_vectorized
        setup = run.setup
        if setup is None or not (setup.has_screen_info() and setup.has_eye_distance()
                                 and setup.has_camera_position()):
            raise ValueError("Foreshortening needs an experimental setup with screen, "
                             "eye distance and camera position (Simulation(setup=...))")
        for eye in run.eyes:
            if eye not in run.x:
                raise ValueError("Foreshortening needs the gaze (e.g., Fixations)")
            x_mm, y_mm = setup.pixels_to_mm(run.x[eye], run.y[eye])
            cos_alpha = _compute_cos_alpha_vectorized(
                x_mm, y_mm, setup.theta, setup.phi, setup.r, setup.d,
                eye_offset=self.eye_offset)
            run.truth[eye, "cosalpha"] = cos_alpha
            run.log_pupil[eye] = run.log_pupil[eye] + 100.0 * np.log(cos_alpha)


@dataclass(repr=False)
class Units(SimStep):
    """
    Units of the tracker: the pupil is :math:`\\text{median} \\cdot
    e^{y / 100}` for the signal :math:`y` in log-%.

    Parameters
    ----------
    median : float or dict
        median pupil size in the units of the tracker (per eye with a
        dict); default 1000

    Notes
    -----
    Ground truth per eye: ``size`` (the true pupil size) and ``pupil`` (the
    pupil as measured without noise, i.e. with the foreshortening).
    """
    median: float = 1000.0
    stage = "units"

    def apply(self, run, rng):
        for eye in run.eyes:
            m = self.median[eye] if isinstance(self.median, dict) else self.median
            run.truth[eye, "size"] = m * np.exp(run.log_size[eye] / 100.0)
            run.pupil[eye] = m * np.exp(run.log_pupil[eye] / 100.0)
            run.truth[eye, "pupil"] = run.pupil[eye].copy()


@dataclass(repr=False)
class WhiteNoise(SimStep):
    """
    White measurement noise on the pupil.

    Parameters
    ----------
    sd : float
        standard deviation of the noise
    scale : {"log%", "data"}
        ``"log%"`` (default): `sd` in log-% (multiplicative noise, like
        the noise of the timescale model); ``"data"``: `sd` in the units of
        the data (additive)
    eyes : str or list, optional
        eyes the step applies to (default: all eyes of the simulation)
    """
    sd: float = 0.2
    scale: str = "log%"
    eyes: tuple = None
    stage = "noise"

    def __post_init__(self):
        super().__post_init__()
        if self.scale not in ("log%", "data"):
            raise ValueError(f"scale must be 'log%' or 'data', got {self.scale!r}")
        if self.sd < 0:
            raise ValueError("sd must not be negative")

    def apply(self, run, rng):
        for eye in self.eyes_in(run):
            e = rng.normal(0.0, self.sd, run.n)
            if self.scale == "log%":
                run.pupil[eye] = run.pupil[eye] * np.exp(e / 100.0)
            else:
                run.pupil[eye] = run.pupil[eye] + e


def _event_times(rate, min_interval, total, rng):
    """onsets (ms) of events at `rate` (per ms) with a minimal interval"""
    if rate <= 0:
        return np.zeros(0)
    mean = 1.0 / rate
    extra = max(mean - min_interval, 1e-9)
    out, t = [], rng.exponential(mean)
    while t < total:
        out.append(t)
        t += min_interval + rng.exponential(extra)
    return np.array(out)


def blink_occlusion(t, tc, tau_c, to, tau_o, a=1.0, w=0.0, tau_s=1.0):
    """
    Occluded fraction of the pupil during a blink (0: open, 1: covered).

    :math:`g(t) = a\\,\\sigma((t - t_c)/\\tau_c)\\,[(1 - w)\\,\\sigma((t_o -
    t)/\\tau_o) + w\\,e^{-\\max(t - t_o, 0)/\\tau_s}]`: the lid closes
    (logistic around :math:`t_c`), and reopens in two speeds: a fast
    logistic around :math:`t_o` and a slow exponential tail with weight
    `w`.

    Parameters
    ----------
    t : array
        times (ms)
    tc, tau_c : float
        midpoint and time scale of the closing (ms)
    to, tau_o : float
        midpoint and time scale of the reopening (ms)
    a : float
        amplitude (1: full blink, < 1: partial)
    w, tau_s : float
        weight and time constant (ms) of the slow tail of the reopening

    Returns
    -------
    np.ndarray
    """
    from scipy.special import expit
    t = np.asarray(t, dtype=float)
    close = expit((t - tc) / tau_c)
    fast = expit((to - t) / tau_o)
    tail = np.exp(-np.clip(t - to, 0, None) / tau_s)
    return a * close * ((1 - w) * fast + w * tail)


@dataclass(repr=False)
class Blinks(SimStep):
    """
    Blinks: the lid covers the pupil, which the tracker loses while the
    covered fraction is above `visibility`.

    The covered fraction :math:`g(t)` of each blink follows
    :func:`blink_occlusion`: a fast closing, a closed phase and a reopening
    with a fast part and a slow tail. The measured pupil is :math:`(1 -
    g)` times the pupil; the samples with :math:`g >` `visibility` are
    missing (with the gaze). The parameters are drawn for each blink; the
    defaults come from fits of the model to blinks of EyeLink recordings
    (grey screen). With several eyes, the blinks are the same in all eyes
    up to a small `jitter`.

    Parameters
    ----------
    rate : str or float
        mean rate (``"15/min"``; plain numbers per minute)
    min_interval : float, str or Quantity
        minimal time between two blinks (default 500 ms); the intervals are
        this plus an exponential time
    closed : float, str or distribution
        time from the closing to the reopening (:math:`t_o - t_c`)
    tau_c, tau_o : float, str or distribution
        time scales of the closing and of the fast reopening
    w : float or distribution
        weight of the slow tail of the reopening
    rho : float or distribution
        time constant of the tail relative to :math:`\\tau_o`
    p_partial : float
        share of partial blinks
    partial : float or distribution
        amplitude of partial blinks (covered fraction at most); partial
        blinks with an amplitude below `visibility` have no missing samples
    visibility : float
        the tracker loses the pupil when more than this fraction is covered
    jitter : float, str or distribution
        shift of each blink in each eye (several eyes)
    eyes : str or list, optional
        eyes (default: all eyes of the simulation)

    Notes
    -----
    Ground truth: ``occlusion`` per eye (the covered fraction :math:`g`),
    and one true event ``"blink"`` per blink and eye in ``d.truth_events``
    (onset :math:`t_c`, with the metadata ``eye``, ``closed``, ``tau_c``,
    ``tau_o``, ``a``, ``w``, ``tau_s`` and ``gap``, the duration of the
    missing samples).
    """
    rate: str = "15/min"
    min_interval: str = "500 ms"
    closed: str = "lognormal(220 ms, 80 ms)"
    tau_c: str = "lognormal(3 ms, 1 ms)"
    tau_o: str = "lognormal(22 ms, 8 ms)"
    w: float = 0.15
    rho: float = 6.0
    p_partial: float = 0.02
    partial: str = "uniform(0.1, 0.4)"
    visibility: float = 0.5
    jitter: str = "normal(0 ms, 2 ms)"
    eyes: tuple = None
    stage = "occlusion"

    def __post_init__(self):
        super().__post_init__()
        from .dist import parse_dist, parse_rate
        parse_rate(self.rate)
        for spec in (self.min_interval, self.closed, self.tau_c, self.tau_o, self.w,
                     self.rho, self.partial, self.jitter):
            parse_dist(spec)
        if not 0 < self.visibility < 1:
            raise ValueError("visibility must be between 0 and 1")
        if not 0 <= self.p_partial <= 1:
            raise ValueError("p_partial must be between 0 and 1")

    def apply(self, run, rng):
        from .dist import parse_dist, parse_rate
        eyes = self.eyes_in(run)
        dt = 1000.0 / run.fs
        t = np.arange(run.n) * dt
        times = _event_times(parse_rate(self.rate), float(parse_dist(self.min_interval).mean),
                             run.n * dt, rng)
        g = {e: np.zeros(run.n) for e in eyes}
        D = {k: parse_dist(getattr(self, k)) for k in
             ("closed", "tau_c", "tau_o", "w", "rho", "partial", "jitter")}
        for tc0 in times:
            p = {k: float(D[k].sample(rng)) for k in ("closed", "tau_c", "tau_o", "w", "rho")}
            p["tau_c"], p["tau_o"] = max(p["tau_c"], 0.1), max(p["tau_o"], 0.1)
            p["closed"], p["rho"] = max(p["closed"], 0.0), max(p["rho"], 1.0)
            p["w"] = min(max(p["w"], 0.0), 1.0)
            partial = rng.random() < self.p_partial
            a = min(max(float(D["partial"].sample(rng)), 0.0), 1.0) if partial else 1.0
            for e in eyes:
                tc = tc0 + (float(D["jitter"].sample(rng)) if len(eyes) > 1 else 0.0)
                to, tau_s = tc + p["closed"], p["rho"] * p["tau_o"]
                lo = np.searchsorted(t, tc - 20 * p["tau_c"] - 20)
                hi = np.searchsorted(t, to + max(10 * p["tau_o"], 8 * tau_s) + 20)
                gk = blink_occlusion(t[lo:hi], tc, p["tau_c"], to, p["tau_o"], a, p["w"], tau_s)
                g[e][lo:hi] = 1 - (1 - g[e][lo:hi]) * (1 - gk)
                run.truth_events.append(dict(
                    onset=tc, label="blink", eye=e, closed=p["closed"], tau_c=p["tau_c"],
                    tau_o=p["tau_o"], a=a, w=p["w"], tau_s=tau_s,
                    gap=float((gk > self.visibility).sum() * dt)))
        for e in eyes:
            run.pupil[e] = run.pupil[e] * (1 - g[e])
            run.truth[e, "occlusion"] = g[e]
            run.lost(e, g[e] > self.visibility)


@dataclass(repr=False)
class TrackerLoss(SimStep):
    """
    Losses of the pupil by the tracker: stretches of missing samples at
    random times, without a transient.

    Parameters
    ----------
    rate : str or float
        mean rate (``"0.5/min"``; plain numbers per minute)
    duration : float, str or distribution
        duration of a loss
    binocular : bool
        if True, the losses are the same in all eyes; if False (default),
        each eye has its own
    eyes : str or list, optional
        eyes (default: all eyes of the simulation)

    Notes
    -----
    Ground truth: one true event ``"loss"`` per loss and eye in
    ``d.truth_events`` (with the metadata ``eye`` and ``duration``).
    """
    rate: str = "0.5/min"
    duration: str = "lognormal(150 ms, 200 ms)"
    binocular: bool = False
    eyes: tuple = None
    stage = "missing"

    def __post_init__(self):
        super().__post_init__()
        from .dist import parse_dist, parse_rate
        parse_rate(self.rate)
        parse_dist(self.duration)

    def apply(self, run, rng):
        from .dist import parse_dist, parse_rate
        eyes = self.eyes_in(run)
        dt = 1000.0 / run.fs
        groups = [eyes] if self.binocular else [(e,) for e in eyes]
        dur = parse_dist(self.duration)
        for group in groups:
            for t0 in _event_times(parse_rate(self.rate), 0.0, run.n * dt, rng):
                length = max(float(dur.sample(rng)), dt)
                a = int(np.ceil(t0 / dt))
                b = min(int(np.ceil((t0 + length) / dt)), run.n)
                if b <= a:
                    continue
                for e in group:
                    mask = np.zeros(run.n, dtype=bool)
                    mask[a:b] = True
                    run.lost(e, mask)
                    run.truth_events.append(dict(onset=a * dt, label="loss", eye=e,
                                                 duration=(b - a) * dt))
