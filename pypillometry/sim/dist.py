"""
Distributions of times, given as strings.

A time in a task (an onset, a duration, the interval between trials, a
reaction time) is either fixed (``"300 ms"``, ``"2 s"``, a plain number in
ms) or drawn from a distribution:

- ``"uniform(2 s, 10 s)"``: uniform between the two values
- ``"normal(500 ms, 30 ms)"``: normal with mean and standard deviation
- ``"lognormal(300 ms, 80 ms)"``: log-normal with mean and standard
  deviation (of the times, not of their logarithm)
- ``"exponential(1 s)"``: exponential with the mean

The values may have units (plain numbers are ms).
"""
import re

import numpy as np

from ..units import parse_time

__all__ = ["parse_dist", "parse_rate"]

_FORM = re.compile(r"^\s*([a-z]+)\s*\((.*)\)\s*$")
_ARGS = {"uniform": 2, "normal": 2, "lognormal": 2, "exponential": 1}


class Dist:
    """
    A distribution of times in ms (see :func:`parse_dist`).

    Attributes
    ----------
    kind : str
        ``"fixed"``, ``"uniform"``, ``"normal"``, ``"lognormal"`` or
        ``"exponential"``
    args : tuple of float
        its parameters in ms
    spec : str or float
        the specification it was made from
    """

    def __init__(self, kind, args, spec):
        self.kind, self.args, self.spec = kind, tuple(float(a) for a in args), spec

    def __repr__(self):
        return f"Dist({self.spec!r})"

    def sample(self, rng, size=None):
        """draw from the distribution (ms)"""
        a = self.args
        if self.kind == "fixed":
            return np.full(size, a[0]) if size is not None else a[0]
        if self.kind == "uniform":
            return rng.uniform(a[0], a[1], size)
        if self.kind == "normal":
            return rng.normal(a[0], a[1], size)
        if self.kind == "lognormal":
            mu, sigma = self._log_params()
            return rng.lognormal(mu, sigma, size)
        return rng.exponential(a[0], size)

    def _log_params(self):
        m, s = self.args
        sigma2 = np.log1p((s / m)**2)
        return np.log(m) - sigma2 / 2, np.sqrt(sigma2)

    @property
    def mean(self):
        """mean (ms)"""
        if self.kind == "uniform":
            return sum(self.args) / 2
        return self.args[0]

    def quantile(self, q):
        """quantile `q` (ms)"""
        from scipy import stats
        a = self.args
        if self.kind == "fixed":
            return a[0]
        if self.kind == "uniform":
            return a[0] + q * (a[1] - a[0])
        if self.kind == "normal":
            return float(stats.norm.ppf(q, a[0], a[1]))
        if self.kind == "lognormal":
            mu, sigma = self._log_params()
            return float(np.exp(mu + sigma * stats.norm.ppf(q)))
        return float(stats.expon.ppf(q, scale=a[0]))

    def __str__(self):
        def ms(v):
            if abs(v) >= 120000:
                return f"{v / 60000:g} min"
            return f"{v / 1000:g} s" if abs(v) >= 1000 else f"{v:g} ms"
        if self.kind == "fixed":
            return ms(self.args[0])
        return f"{self.kind}({', '.join(ms(v) for v in self.args)})"


def parse_dist(spec):
    """
    Distribution of times from its specification.

    Parameters
    ----------
    spec : str, float or Quantity
        a fixed time (``"300 ms"``, ``2.5`` (ms)) or a distribution, e.g.
        ``"uniform(2 s, 10 s)"``, ``"normal(500 ms, 30 ms)"``,
        ``"lognormal(300 ms, 80 ms)"`` (mean and standard deviation of the
        times) or ``"exponential(1 s)"`` (mean)

    Returns
    -------
    Dist
        with ``sample(rng, size)``, ``mean`` and ``quantile(q)``

    Examples
    --------
    >>> import numpy as np
    >>> d = parse_dist("uniform(2 s, 10 s)")
    >>> d.mean, str(d)
    (6000.0, 'uniform(2 s, 10 s)')
    >>> float(parse_dist("300 ms").sample(np.random.default_rng(0)))
    300.0
    """
    if isinstance(spec, Dist):
        return spec
    if isinstance(spec, str):
        m = _FORM.match(spec.lower())
        if m:
            kind, inner = m.groups()
            if kind not in _ARGS:
                raise ValueError(f"unknown distribution {kind!r} in {spec!r}; use "
                                 f"{', '.join(_ARGS)}")
            parts = [p for p in inner.split(",") if p.strip()]
            if len(parts) != _ARGS[kind]:
                raise ValueError(f"{kind} needs {_ARGS[kind]} values, got {spec!r}")
            args = [float(parse_time(p.strip(), to="ms")) for p in parts]
            if kind == "uniform" and args[1] < args[0]:
                raise ValueError(f"uniform: the upper bound is below the lower one ({spec!r})")
            if kind in ("normal", "lognormal") and args[1] < 0:
                raise ValueError(f"{kind}: the standard deviation is negative ({spec!r})")
            if kind in ("lognormal", "exponential") and args[0] <= 0:
                raise ValueError(f"{kind}: the mean must be positive ({spec!r})")
            if kind == "lognormal" and args[1] == 0:
                kind, args = "fixed", args[:1]
            return Dist(kind, args, spec)
    return Dist("fixed", [float(parse_time(spec, to="ms"))], spec)


def parse_rate(spec):
    """
    Rate of events per ms from ``"15/min"``, ``"0.5/s"``, ``"2/h"`` or a
    plain number (per minute).

    Examples
    --------
    >>> parse_rate("15/min") * 60000
    15.0
    """
    if isinstance(spec, str):
        count, sep, unit = spec.partition("/")
        if not sep:
            raise ValueError(f"rate {spec!r}: give it as count/unit, e.g. '15/min'")
        per = float(parse_time(f"1 {unit.strip()}", to="ms"))
        rate = float(count) / per
    else:
        rate = float(spec) / 60000.0
    if rate < 0:
        raise ValueError(f"rate {spec!r} is negative")
    return rate
