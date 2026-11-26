"""
model.py
========

:class:`TimescaleModel`: a state-space model of the pupil size of one
recording with a fast component and slow components at fixed time scales
(see :mod:`pypillometry.signal.gaussmarkov.bank` for the mathematics).
"""
import numpy as np
from loguru import logger

from ..intervals import Intervals
from ..signal import block_mean, semivariogram, spectrum
from ..signal import gaussmarkov as gm
from ..signal.gaussmarkov.bank import DEFAULT_TIMESCALES, _as_prior
from ..units import parse_time
from .plots import TimescalePlots

__all__ = ["TimescaleModel"]


def _dilate(mask, before, after):
    """`mask` with its True runs extended by `before` samples to the left
    and `after` samples to the right"""
    if not mask.any():
        return mask.copy()
    c = np.concatenate([[0], np.cumsum(mask)])
    n = len(mask)
    lo = np.clip(np.arange(n) - after, 0, n)
    hi = np.clip(np.arange(n) + before + 1, 0, n)
    return (c[hi] - c[lo]) > 0


class TimescaleModel:
    """
    Model of the slow and fast fluctuations of the pupil size of one
    recording (experimental).

    The log pupil size is described as a mean per segment, one fast
    component (Matérn-3/2 with free standard deviation and length scale),
    a bank of slow components ("atoms") at fixed time scales whose
    variances are estimated, and white noise. The model answers how much
    the pupil varies at each time scale, returns the slow part of the
    signal (with uncertainty) and reconstructs the signal where it is
    missing.

    The constructor prepares the data and keeps the bookkeeping that is
    needed to get back to the original samples:

    1. the samples that are unusable under the mask policy (blinks have to
       be detected before, see
       :meth:`d.pupil.blinks_detect() <pypillometry.eyedata.pupil.PupilNamespace.blinks_detect>`)
       are extended by `margin`, so that no part of a blink transient stays
       in the data;
    2. the signal is reduced to about `fs` Hz by means of the valid samples
       in consecutive blocks (:func:`~pypillometry.signal.block_mean`;
       nothing is interpolated or filtered);
    3. it is converted to log-%, :math:`100 \\log(\\text{pupil} /
       \\text{median})`, which makes the variability independent of the
       size of the pupil and of its units;
    4. it is split into segments with a mean each: the periods that are not
       excluded (see :meth:`~pypillometry.EyeData.exclude_intervals`), or
       `segments`.

    Parameters
    ----------
    data : EyeData
        the recording, with blinks and artifacts masked
    eyes : str or list
        eyes to model; if empty, all eyes with pupil data
    binocular : bool
        if False (default), each eye gets its own model; if True, one model
        for both eyes: the fast component and the atoms are common to the
        eyes (with a gain between the eyes, which can differ in
        amplitude), each eye has its own mean and noise, and one more
        component describes the difference between the eyes
    timescales : sequence of float, str or Quantity
        time scales of the atoms (plain numbers are ms); default 3, 10, 30,
        100 and 300 s
    segments : Intervals or list of tuple, optional
        independent stretches of the recording with a mean each (e.g.,
        blocks with different luminance), as for
        :meth:`~pypillometry.EyeData.exclude_intervals`; data outside of
        them are not used. Default: the periods that are not excluded
        (:meth:`~pypillometry.EyeData.get_included`).
    fs : float
        sampling rate of the model in Hz (default 20); the rate that is
        used is the sampling rate of the data divided by a whole number
    margin : tuple of float, str or Quantity
        time before and after each run of unusable samples that is not used
        either (plain numbers are ms; default ``(100, 200)``)
    policy : str, MaskPolicy or list of str
        mask policy that decides which samples are usable (see
        :meth:`~pypillometry.EyeData.mask`); default ``"default"``
    variable : str
        variable to model (default ``"pupil"``)
    nu : float
        smoothness of the atoms (default 1.5)
    keep_data : bool
        keep a reference to `data` (default), so that
        :meth:`interpolate` and :meth:`store` work without an argument;
        the model itself only needs the reduced data
    name : str, optional
        name of the model (default: the name of `data`)

    Attributes
    ----------
    eyes : list of str
    units : list of str
        the separately fitted models: the eyes, or ``["both"]`` for a
        binocular model
    bank : TimescaleBank
        structure of the model
    fits : dict
        unit -> :class:`~pypillometry.signal.gaussmarkov.BankFit` (after
        :meth:`fit`)
    fs : float
        sampling rate of the model (Hz)
    time : np.ndarray
        times of the samples of the model (centres of the blocks, ms)
    y : dict
        eye -> the signal the model is fitted to (log-%, NaN where missing)
    segments : list of (start, stop)
        segments in samples of the model
    scale : dict
        eye -> median pupil size (the reference of the log-%)

    Examples
    --------
    >>> import pypillometry as pp
    >>> d = pp.get_example_data("rlmw_002").pupil.blinks_detect()  # doctest: +SKIP
    >>> m = pp.TimescaleModel(d, eyes="left").fit()  # doctest: +SKIP
    >>> m.summary()  # doctest: +SKIP
    >>> m.plot.decomposition()  # doctest: +SKIP
    >>> d = m.interpolate()  # doctest: +SKIP
    """

    def __init__(self, data, eyes=[], binocular=False, timescales=DEFAULT_TIMESCALES,
                 segments=None, fs=20.0, margin=(100, 200), policy="default",
                 variable="pupil", nu=1.5, keep_data=True, name=None):
        data._require(variable)
        self.name = data.name if name is None else name
        self.variable = variable
        self.policy = policy
        self.margin = tuple(float(v) for v in parse_time(margin, to="ms"))
        self.eyes = list(data._eyes_with(eyes, variable))
        self.binocular = bool(binocular)
        if self.binocular and len(self.eyes) != 2:
            raise ValueError("a binocular model needs two eyes with "
                             f"'{variable}' data, got {self.eyes}")
        timescales = [float(parse_time(t, to="ms")) for t in timescales]
        self.bank = gm.TimescaleBank(timescales, nu=nu,
                                     channels=tuple(self.eyes) if self.binocular else 1)
        self.units = ["both"] if self.binocular else list(self.eyes)
        self._unit_eyes = ({"both": tuple(self.eyes)} if self.binocular
                           else {e: (e,) for e in self.eyes})
        self.fits = {}
        self.individual = {}
        self._smoothed = {}
        self._data = data if keep_data else None
        self._prepare(data, segments, fs)

    # -- preparation ---------------------------------------------------------
    def _usable(self, data, eye, margin=None):
        """samples of `eye` in `data` that the model can use (bool array)"""
        margin = self.margin if margin is None else margin
        raw = np.asarray(data.data[eye, self.variable], dtype=float)
        with np.errstate(invalid="ignore"):
            bad = np.asarray(data.mask(eye, self.variable, self.policy), dtype=bool) \
                | ~np.isfinite(raw) | (raw <= 0)
        before, after = (int(np.ceil(v * data.fs / 1000.0)) for v in margin)
        return ~_dilate(bad, before, after)

    def _prepare(self, data, segments, fs):
        n = len(data)
        self._n_samples, self._fs_data, self._t0 = n, float(data.fs), float(data.tx[0])
        k = max(int(round(data.fs / fs)), 1)
        self.factor, self.fs = k, data.fs / k
        if segments is None:
            spans = data.get_included("samples").intervals
        else:
            spans = data._as_spans(segments).to_units("samples", data).intervals
        spans = sorted((max(int(round(a)), 0), min(int(round(b)), n)) for a, b in spans)
        spans = [(a, b) for a, b in spans if b > a]
        if any(b > a2 for (_, b), (a2, _) in zip(spans, spans[1:])):
            raise ValueError("segments must not overlap")
        if not spans:
            raise ValueError("no data: all of the recording is excluded")

        dt = 1000.0 / data.fs
        ys = {e: [] for e in self.eyes}
        ns = {e: [] for e in self.eyes}
        raw, usable = {}, {}
        for e in self.eyes:
            raw[e] = np.asarray(data.data[e, self.variable], dtype=float)
            usable[e] = self._usable(data, e)
            strict = np.asarray(data.mask(e, self.variable, "strict"), dtype=bool)
            if (usable[e] & strict).any():
                logger.warning(
                    f"{self.name}: '{e}_{self.variable}' has interpolated samples that "
                    "the model treats as data; build the model before interpolating, "
                    "or use policy='strict' to leave them out")
            if not usable[e].any():
                raise ValueError(f"'{e}_{self.variable}' has no usable samples")
        self.scale = {e: float(np.median(raw[e][usable[e]])) for e in self.eyes}
        kept, bins, times, start = [], [], [], 0
        for a, b in spans:
            nb = -(-(b - a) // k)       # blocks; the last one may be incomplete
            parts = {}
            for e in self.eyes:
                x = np.full(nb * k, np.nan)
                x[:b - a] = np.where(usable[e][a:b], raw[e][a:b], np.nan)
                parts[e] = block_mean(x, k)
            if not any(np.isfinite(parts[e][0]).any() for e in self.eyes):
                logger.warning(f"{self.name}: segment of samples {a}-{b} has no usable "
                               "data and is left out")
                continue
            for e in self.eyes:
                ys[e].append(100.0 * np.log(parts[e][0] / self.scale[e]))
                ns[e].append(parts[e][1])
            kept.append((a, b))
            bins.append((start, start + nb))
            times.append(data.tx[a] + (np.arange(nb) * k + (k - 1) / 2.0) * dt)
            start += nb
        if not kept:
            raise ValueError("no segment with usable data")
        self._spans = kept
        self.segments = bins
        self.time = np.concatenate(times)
        self.y = {e: np.concatenate(ys[e]) for e in self.eyes}
        self.n = {e: np.concatenate(ns[e]) for e in self.eyes}
        longest = max(b - a for a, b in bins) / self.fs
        slowest = self.bank.timescales[-1] / 1000.0
        if longest < 2 * slowest:
            logger.warning(
                f"{self.name}: the longest segment ({longest:.0f} s) is shorter than "
                f"twice the slowest time scale ({slowest:g} s); the data say little "
                "about the slowest atoms, whose variance then follows the prior")

    def _Y(self, unit):
        """the signal of a unit: (T,) for one eye, (T, 2) for both"""
        eyes = self._unit_eyes[unit]
        return self.y[eyes[0]] if len(eyes) == 1 else np.column_stack([self.y[e] for e in eyes])

    def _where(self, eye):
        """unit and index of the channel of an eye"""
        for unit, eyes in self._unit_eyes.items():
            if eye in eyes:
                return unit, eyes.index(eye)
        raise ValueError(f"the model has no eye {eye!r}; eyes: {self.eyes}")

    def _eyes(self, eyes):
        if isinstance(eyes, str):
            eyes = [eyes]
        eyes = list(eyes) if len(eyes) else list(self.eyes)
        for e in eyes:
            self._where(e)
        return eyes

    def _eye(self, eye):
        """one eye for the plots of a single eye (default: the first)"""
        return self.eyes[0] if eye is None else self._eyes(eye)[0]

    def _cutoff(self, cutoff):
        return float(parse_time(cutoff, to="ms"))

    def __repr__(self):
        ts = ", ".join(self.bank.atom_names)
        dur = sum(b - a for a, b in self.segments) / self.fs / 60.0
        valid = {e: float(np.isfinite(self.y[e]).mean()) for e in self.eyes}
        valid = ", ".join(f"{e} {v:.0%}" for e, v in valid.items())
        state = "fitted" if self.fits else "not fitted"
        return (f"TimescaleModel({self.name!r}, {'binocular, ' if self.binocular else ''}"
                f"eyes={self.eyes}, timescales=({ts}), {len(self.segments)} segment(s), "
                f"{dur:.1f} min at {self.fs:g} Hz, valid: {valid}; {state})")

    # -- fitting -------------------------------------------------------------
    def fit(self, prior="medium", hessian=True, maxiter=400):
        """
        Estimate the parameters of the model.

        Maximises the restricted likelihood of the parameters (Kalman
        filter; the means of the segments are integrated out) plus the log
        density of the prior.

        Parameters
        ----------
        prior : str, GaussianPrior or None
            - ``"weak"``, ``"medium"`` (default) or ``"strong"``: the
              general prior, which describes how the parameters vary
              between recordings of several experiments (see
              :func:`~pypillometry.signal.gaussmarkov.general_prior`)
            - a :class:`~pypillometry.signal.gaussmarkov.GaussianPrior`,
              e.g. ``group.prior`` of a :class:`TimescaleGroup`
            - None: maximum likelihood. Atoms without variance then stay
              at zero, and the split of the variance between neighbouring
              atoms follows the noise of the recording.
        hessian : bool
            compute the Hessian at the estimate (for standard errors);
            default True
        maxiter : int
            maximum number of iterations of the optimiser

        Returns
        -------
        TimescaleModel
            the model itself (fitted)
        """
        for unit in self.units:
            logger.info(f"{self.name}: fitting the timescale model of {unit} "
                        f"({np.isfinite(self._Y(unit)).sum()} valid samples at "
                        f"{self.fs:g} Hz)")
            fit = gm.fit_bank(self.bank, self._Y(unit), self.fs, segments=self.segments,
                              prior=prior, hessian=hessian, maxiter=maxiter)
            if not fit.success:
                logger.warning(f"{self.name}, {unit}: the optimiser did not converge "
                               f"({fit.message})")
            self._set_fit(unit, fit)
        return self

    def _set_fit(self, unit, fit):
        self.fits[unit] = fit
        self._smoothed.pop(unit, None)

    def _fit(self, unit):
        if unit not in self.fits:
            raise ValueError("the model is not fitted yet; call fit() first")
        return self.fits[unit]

    @property
    def params(self):
        """
        Estimated parameters: unit -> dict with the standard deviation
        (log-%) and length scale (ms) of the fast component (``fast_sd``,
        ``fast_ell``), the standard deviation of the noise (``noise_sd``)
        and of each atom (``sd_3s``, ...); see
        :meth:`~pypillometry.signal.gaussmarkov.TimescaleBank.params`.
        """
        return {u: self._fit(u).params for u in self.units}

    @property
    def loglik(self):
        """unit -> restricted log-likelihood at the estimate (without the prior)"""
        return {u: self._fit(u).loglik for u in self.units}

    @property
    def aic(self):
        """unit -> Akaike information criterion (see :attr:`BankFit.aic
        <pypillometry.signal.gaussmarkov.BankFit.aic>`); lower is better"""
        return {u: self._fit(u).aic for u in self.units}

    @property
    def bic(self):
        """unit -> Bayesian information criterion (see :attr:`BankFit.bic
        <pypillometry.signal.gaussmarkov.BankFit.bic>`); lower is better"""
        return {u: self._fit(u).bic for u in self.units}

    def slow_sd(self, cutoff="10 s"):
        """
        Total standard deviation (log-%) of the atoms with a time scale of
        at least `cutoff`: how much the pupil varies slowly. A single
        recording determines this sum much better than its split between
        neighbouring atoms.

        Returns
        -------
        dict
            unit -> float
        """
        cutoff = self._cutoff(cutoff)
        return {u: self.bank.slow_sd(self._fit(u).theta, cutoff) for u in self.units}

    def summary(self, cutoff="10 s"):
        """
        Table of the estimates.

        Parameters
        ----------
        cutoff : float, str or Quantity
            time scale from which on the atoms count as slow (plain numbers
            are ms; default 10 s)

        Returns
        -------
        pandas.DataFrame
            one row per unit: the parameters (see :attr:`params`), the
            total slow standard deviation (``slow_sd``), the number of
            valid samples, the log-likelihood, AIC and BIC
        """
        import pandas as pd
        rows = {}
        for u in self.units:
            fit = self._fit(u)
            rows[u] = {**fit.params, "slow_sd": self.bank.slow_sd(fit.theta, self._cutoff(cutoff)),
                       "n_obs": fit.n_obs, "loglik": fit.loglik, "aic": fit.aic,
                       "bic": fit.bic}
        return pd.DataFrame.from_dict(rows, orient="index")

    # -- smoothing -----------------------------------------------------------
    def _smooth(self, unit):
        if unit not in self._smoothed:
            self._smoothed[unit] = self.bank.smooth(self._fit(unit).theta, self._Y(unit),
                                                    self.fs, self.segments)
        return self._smoothed[unit]

    def components(self, eyes=[], cutoff="10 s"):
        """
        Smoothed components at the sampling rate of the model, in log-%.

        Parameters
        ----------
        eyes : str or list
            eyes; if empty, all eyes of the model
        cutoff : float, str or Quantity
            time scale from which on the atoms count as slow (plain numbers
            are ms; default 10 s)

        Returns
        -------
        dict
            eye -> dict of name -> ``(mean, sd)``, the posterior mean and
            standard deviation of each part at the times :attr:`time`. The
            parts are the components (``"mean"``, ``"fast"``, the atoms
            ``"3s"``, ``"10s"``, ..., and ``"diff"`` for a binocular model)
            and the sums ``"slow"`` (mean plus atoms from `cutoff`),
            ``"detrended"`` (everything else) and ``"signal"`` (everything
            except the noise).
        """
        cutoff = self._cutoff(cutoff)
        out = {}
        for eye in self._eyes(eyes):
            unit, ch = self._where(eye)
            _, sres = self._smooth(unit)
            names = ["mean", "fast"] + (["diff"] if self.binocular else []) \
                + self.bank.atom_names + ["slow", "detrended", "signal"]
            parts = {}
            for name in names:
                w = self.bank.select(name, cutoff, channel=ch, theta=self._fit(unit).theta)
                var = np.einsum("i,tij,j->t", w, sres.P_smooth, w)
                parts[name] = (sres.m_smooth @ w, np.sqrt(np.maximum(var, 0.0)))
            out[eye] = parts
        return out

    def innovations(self, eyes=[]):
        """
        Standardised one-step prediction errors of the model (standard
        normal and uncorrelated if the model is right).

        Returns
        -------
        dict
            eye -> array at the times :attr:`time` (NaN where missing; the
            first sample of each segment, which only determines the mean,
            is NaN too)
        """
        out = {}
        for eye in self._eyes(eyes):
            unit, ch = self._where(eye)
            fres, _ = self._smooth(unit)
            z = fres.std_innov[:, ch].copy()
            for a, b in self.segments:
                valid = np.flatnonzero(np.isfinite(z[a:b]))
                if len(valid):
                    z[a + valid[0]] = np.nan
            out[eye] = z
        return out

    # -- back to the original samples ------------------------------------------
    def _check_data(self, data):
        data = self._data if data is None else data
        if data is None:
            raise ValueError("the model keeps no reference to the data "
                             "(keep_data=False); pass the EyeData object")
        if len(data) != self._n_samples or not np.isclose(data.fs, self._fs_data) \
                or not np.isclose(data.tx[0], self._t0):
            raise ValueError("the data do not have the samples the model was built "
                             "from (the number of samples, the sampling rate or the "
                             "start differ, e.g. after cropping or downsampling)")
        return data

    def _upsampled(self, eye, weights, samples=None):
        """
        Posterior mean and variance (log-%) of parts of the signal of an eye
        at the original samples: (parts, samples of the data). NaN outside
        the segments, in segments without data of the eye and (to save
        time) outside the range of sample indices `samples`.
        """
        unit, _ = self._where(eye)
        _, sres = self._smooth(unit)
        W = np.atleast_2d(weights)
        k = self.factor
        lo, hi = (0, self._n_samples) if samples is None else samples
        # the blocks that cover the range, with one more on each side so
        # that the first and last block have their real neighbours
        todo = []
        for (a, b), (s0, s1) in zip(self.segments, self._spans):
            if s1 <= lo or s0 >= hi or not np.isfinite(self.y[eye][a:b]).any():
                continue
            ba = max(a + (max(lo, s0) - s0) // k - 1, a)
            bb = min(a + -(-(min(hi, s1) - s0) // k) + 1, b)
            todo.append((ba, bb, a, b, s0, s1))
        out_m = np.full((len(W), self._n_samples), np.nan)
        out_v = np.full((len(W), self._n_samples), np.nan)
        if not todo:
            return out_m, out_v
        mean, var = self.bank.upsample(self._fit(unit).theta, self.fs, sres, k, W,
                                       segments=[t[:2] for t in todo])
        for ba, bb, a, b, s0, s1 in todo:
            # the samples of the blocks ba..bb, without the half blocks at
            # the ends that are not ends of the segment (they lack a neighbour)
            first = s0 + (ba - a) * k
            r0 = first + (k // 2 if ba > a else 0)
            r1 = min(s0 + (bb - a) * k - (k // 2 if bb < b else 0), s1)
            i0 = ba * k + (r0 - first)
            out_m[:, r0:r1] = mean[:, i0:i0 + r1 - r0]
            out_v[:, r0:r1] = var[:, i0:i0 + r1 - r0]
        return out_m, out_v

    def predict(self, what="signal", eyes=[], cutoff="10 s", scale="original",
                interval=None, _samples=None):
        """
        Parts of the signal at the original samples.

        The smoother runs at the sampling rate of the model; the model is
        defined in continuous time, so the posterior of the signal at the
        original samples follows exactly from the smoothed states
        (:meth:`~pypillometry.signal.gaussmarkov.TimescaleBank.upsample`).

        Parameters
        ----------
        what : str or list of str
            - ``"signal"`` (default): the pupil size without the noise
              (all components), also where the data are missing
            - ``"slow"``: the mean and the atoms with a time scale of at
              least `cutoff` (the slowly changing level of the pupil)
            - ``"detrended"``: the signal minus its slow part
            - the name of a component (``"mean"``, ``"fast"`` for the fast
              component alone, ``"3s"``, ..., ``"diff"``) or a list of them
        eyes : str or list
            eyes; if empty, all eyes of the model
        cutoff : float, str or Quantity
            time scale from which on the atoms count as slow (plain numbers
            are ms; default 10 s)
        scale : {"original", "log%"}
            ``"original"`` (default): in the units of the data. Parts with
            the mean (signal, slow) are ``median * exp(part / 100)``;
            parts without it are the difference between the signal and
            the signal without the part, so that slow + detrended =
            signal.
            ``"log%"``: on the scale of the model, :math:`100
            \\log(\\text{pupil} / \\text{median})`, where the parts add up.
        interval : float, optional
            if given (e.g. 0.95), also return the limits of the pointwise
            posterior interval with this probability. On the original
            scale only for parts that contain the mean.

        Returns
        -------
        dict
            eye -> array with one value per sample of the data (NaN outside
            the segments); with `interval`, eye -> ``(estimate, lower,
            upper)``
        """
        if scale not in ("original", "log%"):
            raise ValueError(f"scale must be 'original' or 'log%', got {scale!r}")
        cutoff = self._cutoff(cutoff)
        out = {}
        for eye in self._eyes(eyes):
            unit, ch = self._where(eye)
            theta = self._fit(unit).theta
            w = self.bank.select(what, cutoff, channel=ch, theta=theta)
            wsig = self.bank.select("signal", cutoff, channel=ch, theta=theta)
            has_mean = w[ch] != 0
            if scale == "original" and not has_mean:
                if interval is not None:
                    raise ValueError("on the original scale, intervals are only "
                                     "available for parts that contain the mean "
                                     "(e.g. 'signal', 'slow'); use scale='log%'")
                (sig, rest), _ = self._upsampled(eye, [wsig, wsig - w], _samples)
                out[eye] = self.scale[eye] * (np.exp(sig / 100.0) - np.exp(rest / 100.0))
                continue
            (m,), (v,) = self._upsampled(eye, [w], _samples)
            back = (lambda x: self.scale[eye] * np.exp(x / 100.0)) if scale == "original" \
                else (lambda x: x)
            if interval is None:
                out[eye] = back(m)
            else:
                from scipy.stats import norm
                z = norm.ppf(0.5 + interval / 2.0) * np.sqrt(v)
                out[eye] = (back(m), back(m - z), back(m + z))
        return out

    def _record(self, obj, call):
        obj.add_to_history({"funcstring": f"TimescaleModel.{call}",
                            "funcname": f"TimescaleModel.{call.split('(')[0]}"})

    def interpolate(self, data=None, eyes=[], store_as=None, margin=None, inplace=None):
        """
        Replace the unusable samples of the whole recording (blinks,
        artifacts, signal losses of any length) by the expected signal
        under the model.

        The samples that are unusable under the mask policy of the model,
        extended by `margin`, are replaced by ``predict("signal")``: the
        expected pupil size given all usable data of the segment. Other
        than the interpolation of single blinks
        (:meth:`d.pupil.blinks_interpolate() <pypillometry.eyedata.pupil.PupilNamespace.blinks_interpolate>`),
        this uses one model for the whole recording, which also bridges
        long gaps (with the slow components). The replaced samples get the
        ``"interpolated"`` layer. Samples outside the segments of the model
        (e.g., excluded periods) are not changed.

        Parameters
        ----------
        data : EyeData, optional
            object to interpolate; default: the data the model was built
            from. It must have the same samples.
        eyes : str or list
            eyes; if empty, all eyes of the model
        store_as : str, optional
            variable the result is stored in; default: the variable of the
            model (``"pupil"``), which is replaced
        margin : tuple, optional
            time before and after each run of unusable samples that is
            replaced too (plain numbers are ms); default: the margin of the
            model, so that exactly the samples that the model did not use
            are replaced
        inplace : bool or None
            if `True`, change `data` in place and return it; if `False`,
            return a changed copy; if `None`, use the setting of the object

        Returns
        -------
        EyeData
            the object with the interpolated data
        """
        data = self._check_data(data)
        obj = data._get_inplace(inplace)
        eyes = self._eyes(eyes)
        store_as = self.variable if store_as is None else store_as
        margin = self.margin if margin is None else tuple(parse_time(margin, to="ms"))
        signal = self.predict("signal", eyes)
        for eye in eyes:
            key, target = f"{eye}_{self.variable}", f"{eye}_{store_as}"
            replace = ~self._usable(obj, eye, margin) & np.isfinite(signal[eye])
            values = np.asarray(obj.data[key], dtype=float).copy()
            values[replace] = signal[eye][replace]
            if target == key:
                obj.data.set_with_mask(key, values, preserve_mask=True)
            else:
                obj.data._set_channel(target, values, obj.data.layers[key])
            obj.data.set_layer(target, "interpolated", replace)
            logger.debug(f"{target}: {replace.sum()} samples replaced by the model")
        self._record(obj, f"interpolate(eyes={eyes},store_as={store_as!r})")
        return obj

    def store(self, what="slow", store_as=None, data=None, eyes=[], cutoff="10 s",
              inplace=None):
        """
        Store a part of the signal as a variable of the data (in the units
        of the data, one value per sample; see :meth:`predict`).

        Parameters
        ----------
        what : str or list of str
            part of the signal: ``"slow"`` (default), ``"signal"`` (the
            smoothed signal), ``"detrended"`` (the signal minus its slow
            part) or component names
        store_as : str, optional
            name of the variable; default ``"pupilslow"``, ``"pupilsmooth"``
            (for ``"signal"``) or ``"pupildetrended"``
        data : EyeData, optional
            object to store the result in; default: the data the model was
            built from. It must have the same samples.
        eyes : str or list
            eyes; if empty, all eyes of the model
        cutoff : float, str or Quantity
            time scale from which on the atoms count as slow (plain numbers
            are ms; default 10 s)
        inplace : bool or None
            if `True`, change `data` in place and return it; if `False`,
            return a changed copy; if `None`, use the setting of the object

        Returns
        -------
        EyeData
            the object with the new variable (e.g., ``d["left", "pupilslow"]``)
        """
        if store_as is None:
            names = {"slow": "slow", "signal": "smooth", "detrended": "detrended"}
            if not isinstance(what, str) or what not in names:
                raise ValueError("store_as is needed for parts other than 'slow', "
                                 "'signal' and 'detrended'")
            store_as = self.variable + names[what]
        data = self._check_data(data)
        obj = data._get_inplace(inplace)
        eyes = self._eyes(eyes)
        for eye, values in self.predict(what, eyes, cutoff=cutoff).items():
            obj.data[eye, store_as] = values
        self._record(obj, f"store({what!r},store_as={store_as!r},eyes={eyes},"
                          f"cutoff={cutoff!r})")
        return obj

    # -- checks --------------------------------------------------------------
    def variogram(self, eyes=[], per_decade=12):
        """
        Semivariogram of the data of the model (log-%, pooled within the
        segments) at about log-spaced lags.

        Returns
        -------
        dict
            eye -> :class:`~pypillometry.signal.Variogram` (lags in ms)
        """
        max_lag = max(b - a for a, b in self.segments) / 2.0 / self.fs * 1000.0
        return {e: semivariogram(self.y[e], self.fs, max_lag=max_lag,
                                 segments=self.segments).at_log_lags(per_decade)
                for e in self._eyes(eyes)}

    def spectrum(self, eyes=[], window="200 s", **kwargs):
        """
        Power spectrum of the data of the model (log-%, Welch's method
        within the segments; see :func:`~pypillometry.signal.spectrum`).

        Returns
        -------
        dict
            eye -> :class:`~pypillometry.signal.Spectrum`
        """
        return {e: spectrum(self.y[e], self.fs, window=window, segments=self.segments,
                            **kwargs) for e in self._eyes(eyes)}

    def variogram_check(self, eyes=[], n_sim=40, rng=None):
        """
        Semivariogram of the data against simulations of the fitted model
        with the gaps of the data (see
        :func:`~pypillometry.signal.gaussmarkov.variogram_check`).

        Returns
        -------
        dict
            eye -> :class:`~pypillometry.signal.gaussmarkov.VariogramCheck`
        """
        out = {}
        for eye in self._eyes(eyes):
            unit, ch = self._where(eye)
            out[eye] = gm.variogram_check(self.bank, self._fit(unit).theta, self._Y(unit),
                                          self.fs, self.segments, n_sim=n_sim,
                                          channel=ch, rng=rng)
        return out

    def forecast_calibration(self, horizons=("50 ms", "200 ms", "1 s", "5 s", "20 s", "60 s"),
                             step="1 s"):
        """
        Accuracy and calibration of forecasts over several horizons.

        From the state given the data up to a time point, the model
        predicts the signal `horizons` later. A model that describes the
        dependence over a horizon gives standardised errors with a
        standard deviation of 1 (below 1: it expects more variability
        than there is, above 1: less).

        Parameters
        ----------
        horizons : sequence of float, str or Quantity
            how far ahead to predict (plain numbers are ms)
        step : float, str or Quantity
            time between the starting points of the forecasts (default 1 s)

        Returns
        -------
        pandas.DataFrame
            one row per eye and horizon: ``horizon`` (s), ``rmse`` (root
            mean squared error, log-%), ``z_sd`` (standard deviation of
            the standardised errors) and ``n`` (number of forecasts)
        """
        import pandas as pd
        hs = np.unique([max(int(round(parse_time(h, to="samples", fs=self.fs))), 1)
                        for h in horizons])
        step = max(int(round(parse_time(step, to="samples", fs=self.fs))), 1)
        rows = []
        for unit in self.units:
            E, Z = self.bank.forecast_errors(self._fit(unit).theta, self._Y(unit), self.fs,
                                             hs, step=step, segments=self.segments)
            for ch, eye in enumerate(self._unit_eyes[unit]):
                for j, h in enumerate(hs):
                    e, z = E[:, j, ch], Z[:, j, ch]
                    ok = np.isfinite(z)
                    rows.append(dict(eye=eye, horizon=h / self.fs,
                                     rmse=float(np.sqrt(np.mean(e[ok]**2))) if ok.any() else np.nan,
                                     z_sd=float(np.std(z[ok])) if ok.any() else np.nan,
                                     n=int(ok.sum())))
        return pd.DataFrame(rows)

    def cross_validate(self, block="60 s", folds=5, refit=True):
        """
        Blocked cross-validation: the log predictive density of held-out
        blocks given the rest of the recording (see
        :func:`~pypillometry.signal.gaussmarkov.blocked_cv`). Unlike the
        likelihood, the score is sensitive to the slow components; use it
        to compare models of the same data (e.g., with different time
        scales).

        Parameters
        ----------
        block : float, str or Quantity
            length of the blocks (plain numbers are ms; default 60 s)
        folds : int
            number of folds (default 5)
        refit : bool
            re-estimate the parameters for each fold (default); takes
            about as long as the fit

        Returns
        -------
        dict
            unit -> score (higher is better)
        """
        return {u: gm.blocked_cv(self.bank, self._fit(u).theta, self._Y(u), self.fs,
                                 self.segments, prior=self._fit(u).prior, block=block,
                                 folds=folds, refit=refit) for u in self.units}

    @property
    def plot(self):
        """plots of the model (:class:`~pypillometry.timescale.plots.TimescalePlots`)"""
        return TimescalePlots(self)
