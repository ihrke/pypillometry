"""
Steps of a :class:`~pypillometry.sim.Simulation`.

Each step is a small dataclass with the parameters of one part of the
simulated recording. A simulation runs its steps in a fixed order of
stages, whatever the order in which they were added:

1. ``"pupil"``: the latent pupil size in log-%, :math:`100 \\log(\\text{size}
   / \\text{median})` (:class:`TimescalePupil`)
2. ``"gaze"``: the gaze position in pixels (:class:`Fixations`)
3. ``"foreshortening"``: the apparent pupil size as seen by the camera
   (:class:`Foreshortening`)
4. ``"units"``: from log-% to the units of the tracker (:class:`Units`)
5. ``"noise"``: measurement noise (:class:`WhiteNoise`)

Each step draws from its own random stream (derived from the seed of the
simulation, its stage and its position in the stage), so that changing one
step leaves the random numbers of the others unchanged.
"""
from dataclasses import dataclass, fields

import numpy as np

from ..units import parse_time

__all__ = ["TimescalePupil", "Fixations", "Foreshortening", "Units", "WhiteNoise"]

#: order in which the stages are run
STAGES = ("pupil", "gaze", "foreshortening", "units", "noise")


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
        cls = {c.__name__: c for c in (TimescalePupil, Fixations, Foreshortening, Units,
                                       WhiteNoise)}.get(d["step"])
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
