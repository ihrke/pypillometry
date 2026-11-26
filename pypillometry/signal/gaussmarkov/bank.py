"""
bank.py
=======

Timescale-bank model for signals with a continuum of slow time scales
(e.g., the log pupil size).

.. math::

    y_t = \\mu_s + f(t) + \\sum_{k=1}^{K} a_k(t) + \\varepsilon_t,
    \\qquad \\varepsilon_t \\sim N(0, \\sigma_\\varepsilon^2)

- :math:`\\mu_s`: the mean of segment s, with a diffuse prior (the
  likelihood of the other parameters is the restricted likelihood)
- :math:`f`: the fast component, a Matérn-3/2 process with free standard
  deviation and length scale
- :math:`a_k`: the atoms, Matérn processes at fixed time scales
  :math:`\\tau_k` (log-spaced, e.g. 3, 10, 30, 100 and 300 s) with free
  variances :math:`v_k`. "Which slow components does the signal have" is
  then "how much variance is there at each time scale".

The variances of the atoms are estimated as :math:`u_k = \\log(v_k + v_0)`
with a small offset :math:`v_0`, so that an atom without variance sits at
the finite value :math:`\\log v_0` (and not at minus infinity).

With two channels (e.g., the two eyes), the fast component and the atoms
are common to both. Each channel has its own mean and noise; the common
part enters with a gain :math:`g^{-1/2}` in the first and :math:`g^{1/2}`
in the second channel (the channels can differ in amplitude), and one more
Matérn-3/2 process :math:`d(t)` (the difference between the channels)
enters as :math:`+d/2` and :math:`-d/2`:

.. math::

    y_{1,t} = \\mu_{1,s} + g^{-1/2} c(t) + d(t)/2 + \\varepsilon_{1,t}, \\qquad
    y_{2,t} = \\mu_{2,s} + g^{1/2} c(t) - d(t)/2 + \\varepsilon_{2,t},
    \\qquad c = f + \\sum_k a_k

Times are in ms.

Experimental: the API may change.
"""
from dataclasses import dataclass

import numpy as np
from loguru import logger

from ..variogram import Variogram
from .components import Matern
from .kalman import (LGSSM, kalman_filter, kalman_loglik, rts_smoother,
                     forecast_errors, simulate)

__all__ = ["TimescaleBank", "GaussianPrior", "BankFit", "fit_bank",
           "general_prior", "numerical_hessian"]

#: time scales of the atoms of the default bank (ms)
DEFAULT_TIMESCALES = (3000., 10000., 30000., 100000., 300000.)

#: fast length scales (ms) offered to the variogram for the starting values
_START_ELLS = (300., 500., 900., 1500.)


def _label(tau):
    """name of a time scale in ms: 3000 -> '3s', 500 -> '0.5s'"""
    return f"{tau / 1000.:g}s"


class TimescaleBank:
    """
    Structure of a timescale-bank model (see :mod:`~pypillometry.signal.gaussmarkov.bank`)
    and the map between its parameter vector and state-space matrices.

    The object holds no data and no parameter values. The parameter vector
    ``theta`` (see :attr:`names`) is unconstrained apart from box bounds
    (:meth:`bounds`): logs of the standard deviations and of the length
    scale, and :math:`u_k = \\log(v_k + v_0)` for the atoms.

    Parameters
    ----------
    timescales : sequence of float
        time scales (length scales) of the atoms in ms, increasing; default
        3, 10, 30, 100 and 300 s
    nu : float
        smoothness of the atoms (1.5 (default) or 0.5, 2.5, ...)
    v0 : float
        offset of the variances of the atoms (signal units squared); atoms
        with a variance far below it are treated alike. Default 0.25, i.e.
        a standard deviation of 0.5, which suits signals in log-%
        (:math:`100 \\log(x / \\mathrm{median})`).
    channels : int or sequence of str
        1 (default) or 2 channels, or their names (e.g. ``("left",
        "right")``)
    kappa : float
        variance of the diffuse prior of the means

    Examples
    --------
    >>> bank = TimescaleBank()
    >>> bank.names
    ['log_fast_sd', 'log_fast_ell', 'log_noise_sd', 'u_3s', 'u_10s', 'u_30s', 'u_100s', 'u_300s']
    >>> theta = bank.theta(fast_sd=8., fast_ell=700., noise_sd=0.2, sd_30s=4.)
    >>> {k: round(float(v), 3) for k, v in bank.params(theta).items()}
    {'fast_sd': 8.0, 'fast_ell': 700.0, 'noise_sd': 0.2, 'sd_3s': 0.0, 'sd_10s': 0.0, 'sd_30s': 4.0, 'sd_100s': 0.0, 'sd_300s': 0.0}
    >>> bank.lgssm(theta, fs=20).dim_state
    13
    >>> TimescaleBank(channels=("left", "right")).names[2:7]
    ['log_noise_sd_left', 'log_noise_sd_right', 'log_diff_sd', 'log_diff_ell', 'log_gain']
    """

    def __init__(self, timescales=DEFAULT_TIMESCALES, nu=1.5, v0=0.25, channels=1,
                 kappa=1e7):
        self.timescales = tuple(float(t) for t in timescales)
        if len(self.timescales) == 0 or any(t <= 0 for t in self.timescales) \
                or list(self.timescales) != sorted(set(self.timescales)):
            raise ValueError("timescales must be positive, increasing and unique (ms)")
        self.nu = float(nu)
        self.v0 = float(v0)
        if self.v0 <= 0:
            raise ValueError("v0 must be positive")
        self.kappa = float(kappa)
        if isinstance(channels, (int, np.integer)):
            channels = ("",) if channels == 1 else tuple(str(i) for i in range(channels))
        self.channels = tuple(channels)
        if len(self.channels) not in (1, 2):
            raise ValueError("the model has one or two channels")
        self._atoms = [Matern(1.0, t, nu=self.nu) for t in self.timescales]
        self._cache = {}

    def __repr__(self):
        ts = ", ".join(_label(t) for t in self.timescales)
        ch = "" if self.n_channels == 1 else f", channels={self.channels}"
        return f"TimescaleBank(timescales=({ts}), nu={self.nu:g}, v0={self.v0:g}{ch})"

    def __eq__(self, other):
        return isinstance(other, TimescaleBank) and (
            (self.timescales, self.nu, self.v0, self.channels, self.kappa)
            == (other.timescales, other.nu, other.v0, other.channels, other.kappa))

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache"] = {}
        return state

    # -- structure -----------------------------------------------------------
    @property
    def n_channels(self):
        return len(self.channels)

    @property
    def K(self):
        """number of atoms"""
        return len(self.timescales)

    @property
    def atom_names(self):
        """names of the atoms, e.g. ``['3s', '10s', ...]``"""
        return [_label(t) for t in self.timescales]

    @property
    def _suffix(self):
        return [f"_{c}" if c else "" for c in self.channels]

    @property
    def names(self):
        """names of the entries of the parameter vector ``theta``"""
        out = ["log_fast_sd", "log_fast_ell"]
        out += [f"log_noise_sd{s}" for s in self._suffix]
        if self.n_channels == 2:
            out += ["log_diff_sd", "log_diff_ell", "log_gain"]
        return out + [f"u_{a}" for a in self.atom_names]

    @property
    def n_params(self):
        return len(self.names)

    @property
    def _iu(self):
        """index of the first atom in theta"""
        return self.n_params - self.K

    @property
    def component_names(self):
        """names of the components in the order of the state"""
        out = [f"mean{s}" for s in self._suffix] + ["fast"]
        if self.n_channels == 2:
            out.append("diff")
        return out + self.atom_names

    @property
    def slices(self):
        """dict of component name to the slice of the state that belongs to it"""
        m, d = self.n_channels, self._atoms[0].dim
        out = {f"mean{s}": slice(i, i + 1) for i, s in enumerate(self._suffix)}
        out["fast"] = slice(m, m + 2)
        i = m + 2
        if m == 2:
            out["diff"] = slice(i, i + 2)
            i += 2
        for a in self.atom_names:
            out[a] = slice(i, i + d)
            i += d
        return out

    @property
    def dim(self):
        """dimension of the state"""
        return self.n_channels + 2 + 2 * (self.n_channels == 2) + self.K * self._atoms[0].dim

    def gains(self, theta=None):
        """
        Gain of the common components in each channel: 1 for one channel,
        :math:`(g^{-1/2}, g^{1/2})` for two (1 without `theta`).
        """
        if self.n_channels == 1 or theta is None:
            return np.ones(self.n_channels)
        half = 0.5 * float(np.asarray(theta, dtype=float)[6])
        return np.exp([-half, half])

    def obs_matrix(self, theta=None):
        """
        Observation matrix, (channels, dim). With two channels it depends
        on the gain between them (parameter vector `theta`; a gain of 1
        without it).
        """
        sl, m = self.slices, self.n_channels
        gains = self.gains(theta)
        H = np.zeros((m, self.dim))
        for c, s in enumerate(self._suffix):
            H[c, sl[f"mean{s}"].start] = 1.0
            H[c, sl["fast"].start] = gains[c]
            for a in self.atom_names:
                H[c, sl[a].start] = gains[c]
        if m == 2:
            H[0, sl["diff"].start] = 0.5
            H[1, sl["diff"].start] = -0.5
        return H

    @property
    def H(self):
        """observation matrix for a gain of 1 (see :meth:`obs_matrix`)"""
        return self.obs_matrix()

    def select(self, what="signal", cutoff=10000., channel=0, theta=None):
        """
        Weights of the state for a part of the signal of one channel.

        Parameters
        ----------
        what : str or list of str
            - ``"signal"``: everything except the noise
            - ``"slow"``: the mean and the atoms with a time scale of at
              least `cutoff`
            - ``"detrended"``: the signal without its slow part (the fast
              component, the atoms below `cutoff` and, with two channels,
              the difference)
            - the name of a component (see :attr:`component_names`, e.g.
              ``"fast"`` for the fast component alone; ``"mean"`` is the
              mean of the channel) or a list of them
        cutoff : float
            time scale in ms from which on the atoms are "slow" (default 10 s)
        channel : int
            index of the channel (default 0)
        theta : array, optional
            parameter vector; needed with two channels for the gain
            between them (a gain of 1 without it)

        Returns
        -------
        np.ndarray
            vector w, so that the part of the signal is ``state @ w``
        """
        H, sl = self.obs_matrix(theta)[channel], self.slices
        mean = f"mean{self._suffix[channel]}"
        slow = [mean] + [a for a, t in zip(self.atom_names, self.timescales)
                         if t >= cutoff]
        if what == "signal":
            names = self.component_names
        elif what == "slow":
            names = slow
        elif what == "detrended":
            names = [n for n in self.component_names if n not in slow
                     and not n.startswith("mean")]
        else:
            names = [what] if isinstance(what, str) else list(what)
            names = [mean if n == "mean" else n for n in names]
            unknown = set(names) - set(self.component_names)
            if unknown:
                raise ValueError(f"unknown components {sorted(unknown)}; use 'signal', "
                                 f"'slow', 'detrended' or any of {self.component_names}")
        w = np.zeros(self.dim)
        for n in names:
            w[sl[n]] = H[sl[n]]
        return w

    # -- parameters ----------------------------------------------------------
    def atom_var(self, theta):
        """variances of the atoms, :math:`v_k = \\max(e^{u_k} - v_0, 0)`"""
        u = np.asarray(theta, dtype=float)[self._iu:]
        return np.maximum(np.exp(u) - self.v0, 0.0)

    def params(self, theta):
        """
        Parameters on their natural scale.

        Returns
        -------
        dict
            ``fast_sd``, ``fast_ell`` (ms), ``noise_sd`` (per channel:
            ``noise_sd_<channel>``), with two channels ``diff_sd``,
            ``diff_ell`` and ``gain``, and the standard deviation of each
            atom (``sd_3s``, ...)
        """
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_params,):
            raise ValueError(f"theta must have length {self.n_params}")
        out = {n[4:]: float(np.exp(v)) for n, v in zip(self.names[:self._iu], theta)}
        out.update({f"sd_{a}": float(np.sqrt(v))
                    for a, v in zip(self.atom_names, self.atom_var(theta))})
        return out

    def theta(self, **params):
        """
        Parameter vector from parameters on their natural scale (the
        inverse of :meth:`params`); atoms that are not given have no
        variance, the other parameters are required.
        """
        nat = [n[4:] for n in self.names[:self._iu]]
        atoms = [f"sd_{a}" for a in self.atom_names]
        unknown = set(params) - set(nat) - set(atoms)
        missing = set(nat) - set(params)
        if unknown or missing:
            raise ValueError(f"unknown parameters {sorted(unknown)}, missing "
                             f"parameters {sorted(missing)}")
        head = [np.log(params[n]) for n in nat]
        u = [np.log(float(params.get(a, 0.0))**2 + self.v0) for a in atoms]
        return np.array(head + u, dtype=float)

    def slow_sd(self, theta, cutoff=10000.):
        """
        Total standard deviation of the atoms with a time scale of at
        least `cutoff` (ms): :math:`\\sqrt{\\sum_{\\tau_k \\ge cutoff} v_k}`.
        """
        v = self.atom_var(theta)
        return float(np.sqrt(sum(vk for vk, t in zip(v, self.timescales) if t >= cutoff)))

    def bounds(self):
        """list of (lower, upper) bounds of the entries of ``theta``"""
        sd, ell = (np.log(1e-2), np.log(1e3)), (np.log(100.), np.log(2500.))
        b = [sd, ell] + [(np.log(1e-3), np.log(50.))] * self.n_channels
        if self.n_channels == 2:
            b += [(np.log(1e-3), np.log(1e3)), (np.log(100.), np.log(20000.)),
                  (np.log(0.25), np.log(4.0))]
        return b + [(np.log(self.v0), np.log(1e5))] * self.K

    # -- second-order properties ----------------------------------------------
    def _components(self, theta, channel=0):
        """(name, component, weight of its variance in the channel)"""
        p = self.params(theta)
        g2 = float(self.gains(theta)[channel])**2
        out = [("fast", Matern(p["fast_sd"], p["fast_ell"], nu=1.5), g2)]
        if self.n_channels == 2:
            out.append(("diff", Matern(p["diff_sd"], p["diff_ell"], nu=1.5), 0.25))
        for a, t in zip(self.atom_names, self.timescales):
            out.append((a, Matern(p[f"sd_{a}"], t, nu=self.nu), g2))
        return out

    def _noise_var(self, theta):
        m = self.n_channels
        return np.exp(2 * np.asarray(theta, dtype=float)[2:2 + m])

    def variogram(self, theta, lags, channel=0):
        """
        Semivariogram of a channel at lags `lags` (ms), with the nugget
        (the mean does not enter).
        """
        lags = np.asarray(lags, dtype=float)
        g = sum(w * c.variogram(lags) for _, c, w in self._components(theta, channel))
        return g + self._noise_var(theta)[channel] * (lags != 0)

    def psd(self, theta, freq, fs=None, channel=0):
        """
        One-sided power spectral density of a channel at frequencies `freq`
        (Hz), in signal units squared per Hz. With the sampling rate `fs`
        (Hz), the white noise of the samples is included.
        """
        s = sum(w * c.psd(freq) for _, c, w in self._components(theta, channel))
        if fs is not None:
            s = s + self._noise_var(theta)[channel] / (fs / 2.0)
        return s

    # -- state-space model -----------------------------------------------------
    def _unit(self, fs):
        """discretisation of the atoms with variance 1 at sampling rate `fs`"""
        if fs not in self._cache:
            A, Q = zip(*(a.discretise(1000.0 / fs) for a in self._atoms))
            self._cache[fs] = (A, Q, tuple(a.ss().Pinf for a in self._atoms))
        return self._cache[fs]

    def lgssm(self, theta, fs, diffuse=True):
        """
        Discrete-time model for sampling rate `fs` (Hz).

        Parameters
        ----------
        theta : array
            parameter vector
        fs : float
            sampling rate in Hz
        diffuse : bool
            if False, the means are fixed at 0 (for simulation)

        Returns
        -------
        LGSSM
        """
        from scipy.linalg import block_diag
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_params,):
            raise ValueError(f"theta must have length {self.n_params}")
        m, dt = self.n_channels, 1000.0 / fs
        As, Qs, Ps = [np.eye(m)], [np.zeros((m, m))], [np.eye(m) * (self.kappa * diffuse)]
        fast = [Matern(np.exp(theta[0]), np.exp(theta[1]), nu=1.5)]
        if m == 2:
            fast.append(Matern(np.exp(theta[4]), np.exp(theta[5]), nu=1.5))
        for c in fast:
            A, Q = c.discretise(dt)
            As.append(A)
            Qs.append(Q)
            Ps.append(c.ss().Pinf)
        uA, uQ, uP = self._unit(fs)
        v = self.atom_var(theta)
        As += list(uA)
        Qs += [vk * Q for vk, Q in zip(v, uQ)]
        Ps += [vk * P for vk, P in zip(v, uP)]
        return LGSSM(A=block_diag(*As), Q=block_diag(*Qs), H=self.obs_matrix(theta),
                     R=np.diag(self._noise_var(theta)), m0=np.zeros(self.dim),
                     P0=block_diag(*Ps))

    def _y(self, y):
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or y.shape[1] != self.n_channels:
            raise ValueError(f"y must have {self.n_channels} channel(s), got shape {y.shape}")
        return y

    def loglik(self, theta, y, fs, segments=None):
        """
        Restricted log-likelihood of the parameters (Kalman filter).

        Parameters
        ----------
        theta : array
            parameter vector
        y : array, (T,) or (T, channels)
            the signal, NaN where missing
        fs : float
            sampling rate of `y` in Hz
        segments : list of (start, stop), optional
            sample indices of independent segments, each with its own
            mean(s); default: one segment

        Returns
        -------
        float
            does not depend on the level of the signal in each segment
        """
        return kalman_loglik(self.lgssm(theta, fs), self._y(y), n_diffuse=1,
                             segments=segments)

    def filter(self, theta, y, fs, segments=None):
        """Kalman filter (:class:`FilterResult`); arguments as for :meth:`loglik`."""
        return kalman_filter(self.lgssm(theta, fs), self._y(y), n_diffuse=1,
                             segments=segments)

    def smooth(self, theta, y, fs, segments=None):
        """
        Kalman filter and smoother; arguments as for :meth:`loglik`.

        Returns
        -------
        FilterResult, SmootherResult
            use ``sres.m_smooth @ bank.select(..., theta=theta)`` for parts
            of the signal
        """
        ssm = self.lgssm(theta, fs)
        fres = kalman_filter(ssm, self._y(y), n_diffuse=1, segments=segments)
        return fres, rts_smoother(ssm, fres)

    def forecast_errors(self, theta, y, fs, horizons, step=20, segments=None):
        """
        Errors of forecasts `horizons` samples ahead (see
        :func:`~pypillometry.signal.gaussmarkov.forecast_errors`).
        """
        return forecast_errors(self.lgssm(theta, fs), self._y(y), horizons, step=step,
                               segments=segments)

    def simulate(self, theta, n, fs, rng=None, return_states=False):
        """
        Simulate `n` samples at `fs` Hz (the means are 0).

        Returns
        -------
        y : (n,) or (n, 2) array
        x : (n, dim) array
            states (if `return_states`)
        """
        return simulate(self.lgssm(theta, fs, diffuse=False), n, rng=rng,
                        return_states=return_states)

    # -- starting values -------------------------------------------------------
    def start(self, vg, vg_diff=None, floor=1e-3):
        """
        Starting values from an empirical semivariogram.

        The variogram of the model is linear in the variances of its
        components, so that for a dictionary of fast length scales and the
        atoms they follow from non-negative least squares on the relative
        errors; this cannot diverge. The variogram is used at about
        log-spaced lags (:meth:`Variogram.at_log_lags`).

        Parameters
        ----------
        vg : Variogram
            of the signal (pooled within segments); with two channels, of
            their mean
        vg_diff : Variogram, optional
            with two channels: of the difference of the channels
        floor : float
            atoms that get no variance start at this fraction of the largest
            semivariance

        Returns
        -------
        np.ndarray
            parameter vector within the bounds
        """
        from scipy.optimize import nnls

        def decompose(vg, comps):
            ok = np.isfinite(vg.gamma) & (vg.gamma > 0) & (vg.npairs > 0)
            lags, g = vg.lags[ok], vg.gamma[ok]
            if len(g) < 3:
                raise ValueError("the variogram has too few lags with valid pairs")
            X = np.column_stack([np.ones(len(lags))] + [c.variogram(lags) for c in comps])
            coef, _ = nnls(X / g[:, None], np.ones(len(g)))
            return coef, g

        def fast_part(vf, g):
            if vf.sum() <= 0:
                return 0.5 * g.max(), 900.0
            return vf.sum(), float(np.exp(np.sum(vf * np.log(_START_ELLS)) / vf.sum()))

        if not isinstance(vg, Variogram):
            raise TypeError("vg must be a Variogram (see semivariogram())")
        if len(vg.lags) > 200:
            vg = vg.at_log_lags()
        nf = len(_START_ELLS)
        fasts = [Matern(1.0, e, nu=1.5) for e in _START_ELLS]
        coef, g = decompose(vg, fasts + self._atoms)
        vfast, ell = fast_part(coef[1:1 + nf], g)
        noise = max(np.sqrt(coef[0]), 0.02 * np.sqrt(g[0]), 0.02)
        vs = np.maximum(coef[1 + nf:], floor * g.max())
        head = [0.5 * np.log(vfast), np.log(ell)] + [np.log(noise)] * self.n_channels
        if self.n_channels == 2:
            vd, elld = 0.1 * vfast, ell
            if vg_diff is not None:
                if len(vg_diff.lags) > 200:
                    vg_diff = vg_diff.at_log_lags()
                cd, gd = decompose(vg_diff, fasts)
                vd, elld = fast_part(cd[1:], gd)
            head += [0.5 * np.log(vd), np.log(elld), 0.0]
        theta = np.array(head + list(np.log(vs + self.v0)))
        lo, hi = np.array(self.bounds()).T
        return np.clip(theta, lo, hi)

    # -- between the samples -----------------------------------------------------
    def _bridge(self, theta, fs, frac):
        """
        Matrices of the state at a fraction `frac` of the step between two
        samples given the states at both: mean ``B1 x0 + B2 x1``, covariance V.
        """
        dt = 1000.0 / fs
        n = self.dim
        B1, B2, V = np.eye(n), np.zeros((n, n)), np.zeros((n, n))
        if frac <= 0:
            return B1, B2, V
        comps = dict((name, c) for name, c, _ in self._components(theta))
        for name, sl in self.slices.items():
            if name not in comps or comps[name].sigma <= 0:
                continue        # constants (and atoms without variance) do not move
            c = comps[name]
            As, Qs = c.discretise(frac * dt)
            Ar, _ = c.discretise((1 - frac) * dt)
            Ad, Qd = c.discretise(dt)
            b2 = np.linalg.solve(Qd, Ar @ Qs).T
            B1[sl, sl], B2[sl, sl] = As - b2 @ Ad, b2
            v = Qs - b2 @ Ar @ Qs
            V[sl, sl] = 0.5 * (v + v.T)
        return B1, B2, V

    def upsample(self, theta, fs, sres, factor, weights, segments=None, n_out=None):
        """
        Posterior mean and variance of parts of the signal at a `factor`
        times higher sampling rate than the data the smoother was run on.

        The model is defined in continuous time, so that the distribution
        of the state between two samples follows exactly from the smoothed
        states at these samples (a Gauss-Markov bridge). Sample t of the
        smoothed data is taken as the centre of the fine samples
        ``t * factor ... (t + 1) * factor - 1`` (block means, see
        :func:`~pypillometry.signal.block_mean`).

        Parameters
        ----------
        theta : array
            parameter vector
        fs : float
            sampling rate (Hz) of the data the smoother was run on
        sres : SmootherResult
            from :meth:`smooth`
        factor : int
            number of fine samples per sample
        weights : array, (dim,) or (q, dim)
            weights of the state for the part(s) of the signal (see
            :meth:`select`; with two channels, pass `theta` to it)
        segments : list of (start, stop), optional
            the segments used in :meth:`smooth`; outside of them the result
            is NaN
        n_out : int, optional
            number of fine samples (default ``T * factor``; longer outputs
            are padded with NaN)

        Returns
        -------
        mean, var : np.ndarray, (T * factor,) or (q, T * factor)
        """
        factor = int(factor)
        W = np.atleast_2d(np.asarray(weights, dtype=float))
        M, P, Plag = sres.m_smooth, sres.P_smooth, sres.P_lag1
        T, n = M.shape
        if segments is None:
            segments = [(0, T)]
        ssm = self.lgssm(theta, fs)
        A, Q, P0 = ssm.A, ssm.Q, ssm.P0
        # backward transition of the stationary prior, x[t-1] | x[t], for
        # each component (constants and atoms without variance do not move)
        Ab, Qb = A.copy(), np.zeros((n, n))
        for name, sl in self.slices.items():
            Pc = P0[sl, sl]
            if name.startswith("mean") or not Pc.any():
                continue
            Ab[sl, sl] = np.linalg.solve(Pc, A[sl, sl] @ Pc).T
            Qb[sl, sl] = Pc - Ab[sl, sl] @ Pc @ Ab[sl, sl].T
        # fine sample j lies between the centres i and i + 1, at fraction fr
        pos = (np.arange(factor) - (factor - 1) / 2.0) / factor
        before = pos < 0
        fracs = np.where(before, pos + 1.0, pos)
        bridges = [self._bridge(theta, fs, f) for f in fracs]
        n_fine = T * factor if n_out is None else int(n_out)
        mean = np.full((len(W), n_fine), np.nan)
        var = np.full((len(W), n_fine), np.nan)
        for a, b in segments:
            L = b - a
            # one more state before and after the segment (prediction)
            Me = np.concatenate([[Ab @ M[a]], M[a:b], [A @ M[b - 1]]])
            Pe = np.concatenate([[Ab @ P[a] @ Ab.T + Qb], P[a:b],
                                 [A @ P[b - 1] @ A.T + Q]])
            # Cov(x[i+1], x[i]) of neighbours in the extended sequence
            Ce = np.concatenate([[P[a] @ Ab.T], Plag[a + 1:b], [A @ P[b - 1]]])
            for j in range(factor):
                B1, B2, V = bridges[j]
                i0 = np.arange(L) + (0 if before[j] else 1)   # index of the left centre
                out = a * factor + np.arange(L) * factor + j
                ok = out < n_fine
                i0, out = i0[ok], out[ok]
                for q, w in enumerate(W):
                    c1, c2 = B1.T @ w, B2.T @ w
                    mean[q, out] = Me[i0] @ c1 + Me[i0 + 1] @ c2
                    var[q, out] = (w @ V @ w
                                   + np.einsum("i,tij,j->t", c1, Pe[i0], c1)
                                   + np.einsum("i,tij,j->t", c2, Pe[i0 + 1], c2)
                                   + 2 * np.einsum("i,tij,j->t", c2, Ce[i0], c1))
        var = np.maximum(var, 0.0)
        if np.ndim(weights) == 1:
            return mean[0], var[0]
        return mean, var


# ---------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------
@dataclass
class GaussianPrior:
    """
    Gaussian prior of the parameter vector of a :class:`TimescaleBank`.

    Attributes
    ----------
    mean : (p,) array
    cov : (p, p) array
        covariance (a vector of standard deviations is turned into a
        diagonal covariance)
    names : list of str
        names of the parameters (see :attr:`TimescaleBank.names`)
    label : str
        description, e.g. ``"general (medium)"``

    Examples
    --------
    >>> prior = GaussianPrior([0., 1.], [1., 2.], names=["a", "b"])
    >>> prior.sd
    array([1., 2.])
    >>> float(prior.logpdf([0., 3.]))
    -0.5
    """
    mean: np.ndarray
    cov: np.ndarray
    names: list = None
    label: str = ""

    def __post_init__(self):
        self.mean = np.asarray(self.mean, dtype=float)
        cov = np.asarray(self.cov, dtype=float)
        self.cov = np.diag(cov**2) if cov.ndim == 1 else cov
        p = len(self.mean)
        if self.cov.shape != (p, p):
            raise ValueError(f"cov must be ({p}, {p}) or a vector of {p} standard deviations")
        if self.names is not None and len(self.names) != p:
            raise ValueError("names must have one entry per parameter")

    @property
    def sd(self):
        """standard deviations (square roots of the diagonal of `cov`)"""
        return np.sqrt(np.diag(self.cov))

    @property
    def precision(self):
        """inverse of the covariance"""
        return np.linalg.inv(self.cov)

    def logpdf(self, theta):
        """log density at `theta` up to a constant, ``-(theta - mean)' cov^-1 (theta - mean) / 2``"""
        d = np.asarray(theta, dtype=float) - self.mean
        return -0.5 * d @ np.linalg.solve(self.cov, d)

    def scaled(self, factor, label=None):
        """copy with all standard deviations multiplied by `factor`"""
        return GaussianPrior(self.mean.copy(), self.cov * factor**2, self.names,
                             self.label if label is None else label)

    def __repr__(self):
        names = self.names or [f"theta{i}" for i in range(len(self.mean))]
        rows = "\n".join(f"  {n:<20s} {m:8.3f} ± {s:.3f}"
                         for n, m, s in zip(names, self.mean, self.sd))
        full = "" if np.allclose(self.cov, np.diag(np.diag(self.cov))) else ", full covariance"
        return f"GaussianPrior({self.label}{full}; mean ± sd)\n{rows}"


#: Distribution of the parameters over recordings (mean, standard deviation)
#: of log fast sd (log-%), log fast length scale (ms), log noise sd and, at
#: the time scales `_PRIOR_TAUS` (ms), of u = log(v + 0.25); with two
#: channels also of the log sd and length scale of the difference component
#: and of the log gain.
#:
#: Derived in November 2026 from 155 recordings of four experiments with
#: EyeLink trackers (desk-mounted and portable; 500 and 1000 Hz; tasks of 20
#: to 47 minutes): one eye per recording, task periods only, blinks masked
#: with thresholds per recording and margins of 100/200 ms, 20-Hz block
#: means in log-%. A group fit (:func:`fit_group`, diagonal covariance) per
#: experiment gave the mean and the variance between recordings; the values
#: here are the mean of the four group means and the root of the mean
#: within-experiment variance plus the variance of the group means (each
#: experiment has the same weight). The parameters of the difference
#: component and the gain come from binocular fits of 10 recordings per
#: experiment.
_PRIOR_TAUS = (3000., 10000., 30000., 100000., 300000.)
_PRIOR_HEAD = {"log_fast_sd": (2.09, 0.36), "log_fast_ell": (6.52, 0.30),
               "log_noise_sd": (-1.68, 0.53),
               "log_diff_sd": (1.09, 0.37), "log_diff_ell": (6.15, 0.36),
               "log_gain": (0.0, 0.07)}
_PRIOR_U = ((2.42, 3.24, 1.46, 1.61, 2.62), (1.91, 1.89, 1.82, 2.34, 2.66))
#: standard deviations of the general priors relative to the distribution
#: over recordings
_PRIOR_STRENGTH = {"strong": 0.5, "medium": 1.0, "weak": 2.0}


def general_prior(bank=None, strength="medium"):
    """
    General prior for the pupil size in log-% (:math:`100 \\log(x /
    \\mathrm{median})`), derived from recordings of several experiments.

    The prior describes how the parameters of the timescale-bank model
    vary between recordings: 155 recordings of four experiments with
    desk-mounted and portable EyeLink trackers (tasks of 20 to 47 minutes).
    Its centre is a fast component with a standard deviation of 8 log-% and
    a length scale of 0.7 s, noise of 0.2 log-%, and atoms with standard
    deviations of 3.3, 5.0, 2.0, 2.2 and 3.7 log-% at 3, 10, 30, 100 and
    300 s. The fast component and the noise vary little between recordings
    (by a factor of 1.3 to 1.7), the variances of the atoms a lot
    (standard deviations of :math:`u = \\log(v + v_0)` of 1.8 to 2.7).

    The data of a recording determine the fast component, the noise and the
    total slow variance, so that the prior matters for what they do not
    determine: the variance of atoms that carry little of it, and the
    split of the variance between neighbouring atoms. It keeps these in a
    plausible range instead of at zero or at very large values.

    Parameters
    ----------
    bank : TimescaleBank, optional
        the model; default: the default bank. For time scales other than
        the default ones, the prior of the atoms is interpolated over the
        log time scale.
    strength : {"weak", "medium", "strong"} or float
        - ``"medium"`` (default): the distribution of the parameters over
          the recordings
        - ``"strong"``: the same with half the standard deviations
        - ``"weak"``: with twice the standard deviations
        - a number: with the standard deviations multiplied by it

    Returns
    -------
    GaussianPrior
        independent normal distributions of the entries of ``theta``

    Examples
    --------
    >>> prior = general_prior(strength="weak")
    >>> prior.names[:3]
    ['log_fast_sd', 'log_fast_ell', 'log_noise_sd']
    >>> bool(np.allclose(prior.sd, 4 * general_prior(strength="strong").sd))
    True
    """
    bank = TimescaleBank() if bank is None else bank
    if isinstance(strength, str):
        if strength not in _PRIOR_STRENGTH:
            raise ValueError(f"strength must be one of {list(_PRIOR_STRENGTH)} or a "
                             f"number, got {strength!r}")
        factor, label = _PRIOR_STRENGTH[strength], f"general, {strength}"
    else:
        factor, label = float(strength), f"general, sd x {float(strength):g}"
    if abs(bank.v0 - 0.25) > 1e-12:
        logger.warning("the general prior was derived for v0 = 0.25 (signals in log-%)")
    mean, sd = [], []
    for name in bank.names[:bank._iu]:
        key = name if name in _PRIOR_HEAD else "log_noise_sd"
        mean.append(_PRIOR_HEAD[key][0])
        sd.append(_PRIOR_HEAD[key][1])
    lt = np.log(bank.timescales)
    mean += list(np.interp(lt, np.log(_PRIOR_TAUS), _PRIOR_U[0]))
    sd += list(np.interp(lt, np.log(_PRIOR_TAUS), _PRIOR_U[1]))
    return GaussianPrior(np.array(mean), factor * np.array(sd), list(bank.names), label)


def _as_prior(prior, bank):
    """GaussianPrior for `bank` from a prior, the name of a general prior or None"""
    if prior is None:
        return None
    if isinstance(prior, str):
        return general_prior(bank, prior)
    if not isinstance(prior, GaussianPrior):
        raise TypeError("prior must be a GaussianPrior, 'weak', 'medium', 'strong' or None")
    if len(prior.mean) != bank.n_params or (prior.names is not None
                                            and list(prior.names) != bank.names):
        raise ValueError(f"the prior is for the parameters {prior.names}, the model "
                         f"has {bank.names}")
    return prior


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def numerical_hessian(f, x, eps=0.02, mapper=map, callback=None):
    """
    Hessian of `f` at `x` by finite differences with step `eps`.

    The default step suits log-parameters of a Kalman likelihood (with much
    smaller steps, rounding errors dominate). The :math:`1 + 2p + p(p-1)/2`
    evaluations of `f` go through `mapper` (e.g., the ``map`` of a pool of
    threads); `callback` is called after each of them.
    """
    x = np.asarray(x, dtype=float)
    p = len(x)
    E = np.eye(p) * eps
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    points = ([x] + [x + E[i] for i in range(p)] + [x - E[i] for i in range(p)]
              + [x + E[i] + E[j] for i, j in pairs])
    values = []
    for v in mapper(f, points):
        values.append(v)
        if callback is not None:
            callback()
    f0, fp, fm = values[0], values[1:p + 1], values[p + 1:2 * p + 1]
    H = np.zeros((p, p))
    for i in range(p):
        H[i, i] = (fp[i] - 2 * f0 + fm[i]) / eps**2
    for (i, j), fij in zip(pairs, values[2 * p + 1:]):
        H[i, j] = H[j, i] = (fij - fp[i] - fp[j] + f0) / eps**2
    return H


def _n_threads(n_jobs, n_params):
    """number of threads for a fit: one per parameter at most (the points
    of a finite-difference gradient); None means all cores but one"""
    import os
    if n_jobs is None:
        n_jobs = max((os.cpu_count() or 2) - 1, 1)
    return max(min(int(n_jobs), n_params), 1)


def _psd(H):
    """nearest positive semi-definite matrix (negative eigenvalues set to 0)"""
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    return (V * np.maximum(w, 0.0)) @ V.T


@dataclass
class BankFit:
    """
    Result of :func:`fit_bank`.

    Attributes
    ----------
    bank : TimescaleBank
    theta : np.ndarray
        estimate of the parameter vector (the mode of the posterior, or
        the maximum of the likelihood without a prior)
    loglik : float
        restricted log-likelihood at `theta` (without the prior)
    n_obs : int
        number of valid observations
    n_segments : int
        number of segments (each with a mean per channel)
    prior : GaussianPrior or None
    hessian : np.ndarray or None
        Hessian of the negative log-likelihood at `theta` (without the
        prior)
    success : bool
        whether the optimiser converged
    message : str
        message of the optimiser
    """
    bank: TimescaleBank
    theta: np.ndarray
    loglik: float
    n_obs: int
    n_segments: int = 1
    prior: GaussianPrior = None
    hessian: np.ndarray = None
    success: bool = True
    message: str = ""
    nfev: int = 0

    @property
    def params(self):
        """parameters on their natural scale (see :meth:`TimescaleBank.params`)"""
        return self.bank.params(self.theta)

    @property
    def n_params(self):
        """number of fitted parameters (without the means, which are integrated out)"""
        return self.bank.n_params

    @property
    def logpost(self):
        """log-likelihood plus log prior density (up to a constant)"""
        return self.loglik + (0.0 if self.prior is None else self.prior.logpdf(self.theta))

    @property
    def aic(self):
        """
        Akaike information criterion, ``-2 loglik + 2 k`` (lower is
        better), with the restricted log-likelihood at the estimate and
        the number of fitted parameters k. With an informative prior the
        effective number of parameters is smaller than k.
        """
        return -2 * self.loglik + 2 * self.n_params

    @property
    def bic(self):
        """
        Bayesian information criterion, ``-2 loglik + k log(n)`` (lower is
        better); n is the number of valid observations minus the number
        of estimated means.
        """
        n = self.n_obs - self.n_segments * self.bank.n_channels
        return -2 * self.loglik + self.n_params * np.log(max(n, 1))

    @property
    def cov(self):
        """
        Approximate covariance of the estimate (Laplace approximation),
        ``(H + prior precision)^-1`` with the negative eigenvalues of the
        Hessian H set to 0, so that it is never larger than the prior.
        None without a Hessian, or if the matrix is singular (parameters
        that the data do not determine, without a prior).
        """
        if self.hessian is None:
            return None
        P = _psd(self.hessian)
        if self.prior is not None:
            P = P + self.prior.precision
        try:
            return np.linalg.inv(P)
        except np.linalg.LinAlgError:
            return None

    @property
    def se(self):
        """approximate standard errors of `theta` (see :attr:`cov`), or None"""
        cov = self.cov
        return None if cov is None else np.sqrt(np.clip(np.diag(cov), 0, None))


def fit_bank(bank, y, fs, segments=None, prior="medium", theta0=None, hessian=True,
             maxiter=400, n_jobs=1, progress=False, desc="fit"):
    """
    Fit a timescale-bank model to a signal.

    Maximises the restricted log-likelihood (Kalman filter) plus the log
    density of the prior over the parameter vector, within the bounds of
    the model (L-BFGS-B). The starting values come from the semivariogram
    of the signal (:meth:`TimescaleBank.start`).

    Parameters
    ----------
    bank : TimescaleBank
    y : array, (T,) or (T, channels)
        the signal, NaN where missing
    fs : float
        sampling rate of `y` in Hz
    segments : list of (start, stop), optional
        sample indices of independent segments with a mean each; default:
        one segment
    prior : GaussianPrior, str or None
        prior of the parameter vector: a :class:`GaussianPrior` (e.g., from
        a group fit), the strength of the general prior (``"weak"``,
        ``"medium"`` (default) or ``"strong"``, see :func:`general_prior`)
        or None for maximum likelihood
    theta0 : array, optional
        starting values (default: from the variogram)
    hessian : bool
        compute the Hessian of the negative log-likelihood at the estimate
        (for standard errors and group fits); default True
    maxiter : int
        maximum number of iterations of the optimiser
    n_jobs : int or None
        number of threads that evaluate the likelihood at the points of the
        finite-difference gradient and of the Hessian (at most one per
        parameter); None: all cores but one. Default 1 (e.g., inside
        parallel processes).
    progress : bool
        show a progress bar (iterations and the negative log posterior,
        then the evaluations of the Hessian); default False
    desc : str
        label of the progress bar

    Returns
    -------
    BankFit
    """
    from scipy.optimize import minimize
    from ..variogram import semivariogram
    y = bank._y(y)
    T = len(y)
    segments = [(0, T)] if segments is None else [(int(a), int(b)) for a, b in segments]
    prior = _as_prior(prior, bank)
    inside = np.zeros(T, dtype=bool)
    for a, b in segments:
        inside[a:b] = True
    n_obs = int((np.isfinite(y) & inside[:, None]).sum())
    if n_obs < 10 * bank.n_params:
        raise ValueError(f"only {n_obs} valid samples: too few to fit the model")
    ssm_args = dict(n_diffuse=1, segments=segments)

    def loglik(theta):
        try:
            ll = kalman_loglik(bank.lgssm(theta, fs), y, **ssm_args)
        except (np.linalg.LinAlgError, ValueError):
            return -np.inf
        return ll if np.isfinite(ll) else -np.inf

    def neg(theta):
        ll = loglik(theta)
        if prior is not None:
            ll += prior.logpdf(theta)
        return -ll if np.isfinite(ll) else 1e300

    lo, hi = np.array(bank.bounds()).T
    if theta0 is None:
        max_lag = max(b - a for a, b in segments) / 2.0 / fs * 1000.0
        kw = dict(fs=fs, max_lag=max_lag, segments=segments)
        if bank.n_channels == 1:
            theta0 = bank.start(semivariogram(y[:, 0], **kw))
        else:
            theta0 = bank.start(semivariogram(y.mean(axis=1), **kw),
                                semivariogram(y[:, 0] - y[:, 1], **kw))
    theta0 = np.clip(np.asarray(theta0, dtype=float), lo, hi)
    n_threads = _n_threads(n_jobs, bank.n_params)
    pool = bar = None
    try:
        options = dict(maxiter=maxiter)
        if n_threads > 1:
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(max_workers=n_threads)
            options["workers"] = pool.map
        callback = None
        if progress:
            from tqdm import tqdm
            bar = tqdm(desc=desc, unit=" it", leave=False)

            def callback(intermediate_result):
                bar.update()
                bar.set_postfix_str(f"-logpost={intermediate_result.fun:.2f}",
                                    refresh=False)
        opt = minimize(neg, theta0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                       options=options, callback=callback)
        theta = np.clip(opt.x, lo, hi)
        H = None
        if hessian:
            p = bank.n_params
            if bar is not None:
                bar.set_description(f"{desc}: Hessian")
                bar.unit = " evaluations"
                bar.reset(total=1 + 2 * p + p * (p - 1) // 2)
            H = numerical_hessian(lambda th: -loglik(th), theta,
                                  mapper=map if pool is None else pool.map,
                                  callback=None if bar is None else bar.update)
    finally:
        if pool is not None:
            pool.shutdown()
        if bar is not None:
            bar.close()
    if H is not None and not np.all(np.isfinite(H)):
        H = None
    return BankFit(bank, theta, float(loglik(theta)), n_obs, len(segments), prior, H,
                   bool(opt.success), str(opt.message), int(opt.nfev))
