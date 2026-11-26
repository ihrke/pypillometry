"""
variogram.py
============

Empirical (semi-)variogram of a signal with missing samples.

The semivariogram describes how different two samples are on average as a
function of the time between them (the lag):

.. math::

    \\gamma(h) = \\frac{1}{2 N(h)} \\sum_{t:\\, v_t v_{t+h}} (x_{t+h} - x_t)^2

where only pairs of samples that are both valid (:math:`v_t = v_{t+h} = 1`)
enter the sum and :math:`N(h)` is their number. For a stationary signal,
:math:`\\gamma(h)` rises from the noise level (the "nugget") to the variance
of the signal (the "sill"); for a signal with slow drifts it keeps rising.

Experimental: the API may change.
"""
from typing import NamedTuple

import numpy as np
from loguru import logger

from ..units import parse_time

__all__ = ["Variogram", "semivariogram", "plot_variogram"]


class Variogram(NamedTuple):
    """
    Result of :func:`semivariogram`.

    Attributes
    ----------
    lags : np.ndarray
        lags in ms (1, 2, ... samples)
    gamma : np.ndarray
        semivariance at each lag (signal units squared); NaN where there
        are no pairs of valid samples
    npairs : np.ndarray
        number of pairs of valid samples at each lag
    variance : float
        variance of the valid samples (the sill of a stationary signal)
    """
    lags: np.ndarray
    gamma: np.ndarray
    npairs: np.ndarray
    variance: float

    def at_log_lags(self, per_decade=12, min_pairs=200):
        """
        The variogram at about log-spaced lags.

        Picks `per_decade` of the lags per factor of 10 (all lags where
        they are further apart than that), so that fits and plots are not
        dominated by the many long lags.

        Parameters
        ----------
        per_decade : int, optional
            number of lags per decade (default 12)
        min_pairs : int, optional
            lags with fewer pairs of valid samples are left out (default 200)

        Returns
        -------
        Variogram

        Examples
        --------
        >>> x = np.random.default_rng(0).normal(size=5000)
        >>> vg = semivariogram(x, fs=1000, max_lag=1000).at_log_lags(per_decade=4)
        >>> vg.lags
        array([   1.,    2.,    3.,    6.,   10.,   18.,   32.,   56.,  100.,
                178.,  316.,  562., 1000.])
        """
        nlag = len(self.lags)
        idx = np.round(np.logspace(0, np.log10(nlag),
                                   int(per_decade * np.log10(nlag)) + 1))
        idx = np.unique(idx.astype(int)) - 1
        idx = idx[(self.npairs[idx] >= min_pairs) & np.isfinite(self.gamma[idx])]
        return Variogram(self.lags[idx], self.gamma[idx], self.npairs[idx],
                         self.variance)


def _valid_data(signal, mask):
    """Data (float, 0 where invalid) and the boolean validity of each sample."""
    if isinstance(signal, np.ma.MaskedArray):
        data = np.asarray(signal.data, dtype=float)
        invalid = np.ma.getmaskarray(signal)
    else:
        data = np.asarray(signal, dtype=float)
        invalid = np.zeros(data.shape, dtype=bool)
    if data.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if mask is not None:
        invalid = invalid | np.asarray(mask, dtype=bool)
    valid = ~invalid & np.isfinite(data)
    return np.where(valid, data, 0.0), valid


def _variogram_direct(x, v, nlag):
    """Sum of squared differences and number of pairs, one lag at a time."""
    ssd = np.zeros(nlag)
    npairs = np.zeros(nlag)
    for h in range(1, nlag + 1):
        ok = v[h:] & v[:-h]
        d = x[h:][ok] - x[:-h][ok]
        ssd[h - 1] = np.dot(d, d)
        npairs[h - 1] = ok.sum()
    return ssd, npairs


def _variogram_fft(x, v, nlag):
    """
    Sum of squared differences and number of pairs via FFT.

    With z = x*v and w = x^2*v (zero at invalid samples),
    sum_t v_t v_{t+h} (x_{t+h} - x_t)^2
        = sum_t v_t w_{t+h} + sum_t w_t v_{t+h} - 2 sum_t z_t z_{t+h},
    and each sum is a cross-correlation at lag h.
    """
    from scipy.fft import rfft, irfft, next_fast_len
    n = len(x)
    nfft = next_fast_len(n + nlag + 1, real=True)
    vf = v.astype(float)
    V, W, Z = (rfft(a, nfft) for a in (vf, x**2 * vf, x * vf))

    def xcorr(A, B):  # sum_t a_t b_{t+h} for h = 1..nlag
        return irfft(np.conj(A) * B, nfft)[1:nlag + 1]

    ssd = xcorr(V, W) + xcorr(W, V) - 2 * xcorr(Z, Z)
    npairs = np.rint(xcorr(V, V))
    return np.maximum(ssd, 0.0), npairs


def semivariogram(signal, fs, mask=None, max_lag=2000, method="fft", segments=None):
    """
    Empirical semivariogram of a signal with missing samples.

    Only pairs of samples that are both valid are used (see
    :mod:`pypillometry.signal.variogram` for the formula). Invalid samples
    are the masked ones and NaN/inf values; they are neither interpolated
    nor dropped, so gaps do not shift the time axis.

    Parameters
    ----------
    signal : np.ndarray or np.ma.MaskedArray
        the signal, sampled at `fs`; the mask of a masked array is used
    fs : float
        sampling rate in Hz
    mask : np.ndarray, optional
        boolean array, True for invalid samples (combined with the mask of
        a masked array and with NaN values)
    max_lag : float, str or Quantity, optional
        largest lag; plain numbers are ms (default 2000 ms). Limited to
        the length of the signal.
    method : {"fft", "direct"}, optional
        "fft" (default) computes all lags at once in O(n log n); "direct"
        loops over the lags in O(n * lags) and is exact (used to check
        the FFT version).
    segments : list of (start, stop), optional
        sample indices of segments (e.g., blocks of an experiment): only
        pairs of samples within the same segment are used, each segment
        around its own mean, and the sums are pooled over the segments.
        This keeps changes of the level between the segments out of the
        variogram. Samples outside the segments are not used.

    Returns
    -------
    Variogram
        named tuple ``(lags, gamma, npairs, variance)``, lags in ms

    Examples
    --------
    White noise has a flat variogram at its variance:

    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(0, 2, 10000)
    >>> x[1000:1200] = np.nan   # a gap
    >>> vg = semivariogram(x, fs=1000, max_lag=50)
    >>> vg.lags[:3]
    array([1., 2., 3.])
    >>> bool(np.allclose(vg.gamma, 4, rtol=0.1))
    True
    """
    if method not in ("fft", "direct"):
        raise ValueError(f"method must be 'fft' or 'direct', got '{method}'")
    x, v = _valid_data(signal, mask)
    n = len(x)
    if segments is None:
        segments = [(0, n)]
    else:
        segments = [(max(int(a), 0), min(int(b), n)) for a, b in segments]
        segments = [(a, b) for a, b in segments if b > a]
    longest = max([b - a for a, b in segments], default=1)
    nlag = int(round(parse_time(max_lag, to="samples", fs=fs)))
    nlag = min(max(nlag, 1), max(longest - 1, 1))
    lags = np.arange(1, nlag + 1) / fs * 1000.

    used = np.zeros(n, dtype=bool)
    for a, b in segments:
        used[a:b] = v[a:b]
    if not used.any():
        logger.warning("No valid samples: variogram is all NaN")
        nan = np.full(nlag, np.nan)
        return Variogram(lags, nan, np.zeros(nlag), np.nan)

    fct = _variogram_fft if method == "fft" else _variogram_direct
    ssd, npairs, ss = np.zeros(nlag), np.zeros(nlag), 0.0
    for a, b in segments:
        xs, vs = x[a:b], v[a:b]
        if not vs.any():
            continue
        # the variogram does not depend on the mean; removing it keeps the
        # FFT sums small (less cancellation)
        xs = np.where(vs, xs - xs[vs].mean(), 0.0)
        ss += float(np.sum(xs**2))
        m = min(nlag, b - a - 1)
        if m > 0:
            s_, n_ = fct(xs, vs, m)
            ssd[:m] += s_
            npairs[:m] += n_
    variance = ss / used.sum()

    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = np.where(npairs > 0, ssd / (2 * npairs), np.nan)
    return Variogram(lags, gamma, npairs.astype(int), variance)


def plot_variogram(vg, ax=None, scale="loglog", normalize=False,
                   min_pairs=100, show_sill=True, units="ms", **kwargs):
    """
    Plot one or several variograms.

    Parameters
    ----------
    vg : Variogram or dict of Variogram
        result(s) of :func:`semivariogram`; the keys of a dict are the labels
        (e.g., ``{"left_pupil": ..., "right_pupil": ...}``)
    ax : matplotlib Axes, optional
        axes to draw on; default: a new figure
    scale : {"loglog", "logx", "linear"}, optional
        axis scaling; default "loglog" (power laws are straight lines)
    normalize : bool, optional
        divide by the variance of the signal, so that the sill of a
        stationary signal is at 1; default False
    min_pairs : int, optional
        lags with fewer pairs of valid samples are drawn faded; default 100
    show_sill : bool, optional
        draw the variance of each signal as a dashed line; default True
    units : str, optional
        unit of the lag axis ("ms", "sec", "min"); default "ms"
    **kwargs
        passed to ``ax.plot``

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt
    if scale not in ("loglog", "logx", "linear"):
        raise ValueError(f"scale must be 'loglog', 'logx' or 'linear', got '{scale}'")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    vgs = {None: vg} if isinstance(vg, Variogram) else dict(vg)

    for label, g in vgs.items():
        lags = parse_time(g.lags, to=units)
        gamma = g.gamma / g.variance if normalize else g.gamma
        reliable = g.npairs >= min_pairs
        line, = ax.plot(lags, np.where(reliable, gamma, np.nan),
                        label=label, **kwargs)
        if not reliable.all():
            ax.plot(lags, np.where(reliable, np.nan, gamma),
                    color=line.get_color(), alpha=0.3, lw=line.get_linewidth())
        if show_sill:
            ax.axhline(1.0 if normalize else g.variance, ls="--", lw=0.8,
                       color=line.get_color(), alpha=0.7)

    if scale in ("loglog", "logx"):
        ax.set_xscale("log")
    if scale == "loglog":
        ax.set_yscale("log")
    ax.set_xlabel(f"lag ({units})")
    ax.set_ylabel("semivariance / variance" if normalize else "semivariance")
    ax.grid(True, which="major", alpha=0.3)
    if any(label is not None for label in vgs):
        ax.legend(frameon=False)
    return ax
