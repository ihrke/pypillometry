"""
spectral.py
===========

Power spectrum of a signal with missing samples (Welch's method).

The signal is cut into overlapping windows; windows with too many missing
samples are skipped, the gaps in the others are bridged linearly, and the
periodograms of the windows are averaged. Windows never extend over the
border of a segment.

Experimental: the API may change.
"""
from typing import NamedTuple

import numpy as np
from loguru import logger

from ..units import parse_time
from .variogram import _valid_data

__all__ = ["Spectrum", "spectrum", "plot_spectrum"]


class Spectrum(NamedTuple):
    """
    Result of :func:`spectrum`.

    Attributes
    ----------
    freqs : np.ndarray
        frequencies in Hz (without 0)
    power : np.ndarray
        one-sided power spectral density (signal units squared per Hz);
        NaN if no window could be used
    nwindows : int
        number of windows that were averaged
    window : float
        length of the windows in ms
    """
    freqs: np.ndarray
    power: np.ndarray
    nwindows: int
    window: float

    def at_log_freqs(self, per_decade=20):
        """
        The spectrum averaged in log-spaced frequency bands.

        A Welch spectrum has equally spaced frequencies, so most of its
        points (and most of its scatter) are at the highest frequencies.
        The mean power in `per_decade` bands per factor of 10 gives a
        smoother curve on logarithmic axes; bands without a frequency are
        left out.

        Parameters
        ----------
        per_decade : int, optional
            number of bands per decade (default 20)

        Returns
        -------
        Spectrum
            `freqs` are the geometric means of the frequencies in each band
        """
        f, p = self.freqs, self.power
        if len(f) == 0 or not np.isfinite(p).any():
            return self
        band = np.floor(np.log10(f / f[0]) * per_decade + 0.5).astype(int)
        _, start = np.unique(band, return_index=True)
        cnt = np.diff(np.r_[start, len(f)])
        fm = np.exp(np.add.reduceat(np.log(f), start) / cnt)
        pm = np.add.reduceat(p, start) / cnt
        return Spectrum(fm, pm, self.nwindows, self.window)


def _bridge(x, v):
    """`x` with the invalid samples bridged linearly (ends: nearest valid value)"""
    if v.all():
        return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[v], x[v])


def spectrum(signal, fs, mask=None, window="200 s", overlap=0.5, max_missing=0.3,
             segments=None, detrend="linear", taper="hann"):
    """
    Power spectral density of a signal with missing samples (Welch's method).

    Parameters
    ----------
    signal : np.ndarray or np.ma.MaskedArray
        the signal, sampled at `fs`; masked, NaN and inf samples are missing
    fs : float
        sampling rate in Hz
    mask : np.ndarray, optional
        boolean array, True for additional missing samples
    window : float, str or Quantity, optional
        length of the windows (plain numbers are ms; default 200 s). The
        lowest frequency of the spectrum is one cycle per window. Limited
        to the length of the longest segment.
    overlap : float, optional
        overlap of neighbouring windows as a fraction of their length
        (default 0.5)
    max_missing : float, optional
        windows with a larger fraction of missing samples are skipped
        (default 0.3). The gaps in the windows that are used are bridged
        linearly, which removes power at high frequencies in proportion to
        the amount of missing data.
    segments : list of (start, stop), optional
        sample indices of segments (e.g., blocks of an experiment); windows
        lie within one segment. Default: the whole signal.
    detrend : {"linear", "constant"} or False, optional
        trend removed from each window before the transform (default
        "linear")
    taper : str, optional
        window function from :func:`scipy.signal.get_window` (default "hann")

    Returns
    -------
    Spectrum
        named tuple ``(freqs, power, nwindows, window)``

    Examples
    --------
    White noise with standard deviation 2 at 100 Hz has a flat spectrum of
    ``2**2 / (fs / 2)``:

    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(0, 2, 60000)
    >>> sp = spectrum(x, fs=100, window="10 s")
    >>> sp.nwindows, float(sp.freqs[0])
    (119, 0.1)
    >>> bool(np.isclose(sp.power.mean(), 4 / 50, rtol=0.02))
    True
    """
    from scipy.signal import get_window
    from scipy.signal import detrend as sp_detrend
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    x, v = _valid_data(signal, mask)
    n = len(x)
    if segments is None:
        segments = [(0, n)]
    else:
        segments = [(max(int(a), 0), min(int(b), n)) for a, b in segments]
        segments = [(a, b) for a, b in segments if b > a]
    longest = max([b - a for a, b in segments], default=0)
    nwin = int(round(parse_time(window, to="samples", fs=fs)))
    nwin = min(nwin, longest)
    if nwin < 8:
        raise ValueError("the window is shorter than 8 samples (or there is no data)")
    step = max(int(round(nwin * (1 - overlap))), 1)
    w = get_window(taper, nwin)
    norm = fs * np.sum(w**2)

    freqs = np.fft.rfftfreq(nwin, 1.0 / fs)[1:]
    acc, count = np.zeros(len(freqs)), 0
    for a, b in segments:
        for start in range(a, b - nwin + 1, step):
            vs = v[start:start + nwin]
            if 1 - vs.mean() > max_missing or vs.sum() < 2:
                continue
            xs = _bridge(x[start:start + nwin], vs)
            if detrend:
                xs = sp_detrend(xs, type=detrend)
            acc += np.abs(np.fft.rfft(xs * w)[1:])**2
            count += 1
    if count == 0:
        logger.warning("No window with enough valid samples: spectrum is all NaN")
        return Spectrum(freqs, np.full(len(freqs), np.nan), 0, nwin / fs * 1000.)
    power = 2 * acc / (count * norm)
    if nwin % 2 == 0:
        power[-1] /= 2      # the Nyquist frequency is not doubled
    return Spectrum(freqs, power, count, nwin / fs * 1000.)


def plot_spectrum(sp, ax=None, per_decade=20, scale="loglog", **kwargs):
    """
    Plot one or several spectra.

    Parameters
    ----------
    sp : Spectrum or dict of Spectrum
        result(s) of :func:`spectrum`; the keys of a dict are the labels
    ax : matplotlib Axes, optional
        axes to draw on; default: a new figure
    per_decade : int or None, optional
        average the power in this many log-spaced bands per decade (see
        :meth:`Spectrum.at_log_freqs`; default 20); None: all frequencies
    scale : {"loglog", "logy", "linear"}, optional
        axis scaling; default "loglog" (power laws are straight lines)
    **kwargs
        passed to ``ax.plot``

    Returns
    -------
    matplotlib Axes
    """
    import matplotlib.pyplot as plt
    if scale not in ("loglog", "logy", "linear"):
        raise ValueError(f"scale must be 'loglog', 'logy' or 'linear', got '{scale}'")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    sps = {None: sp} if isinstance(sp, Spectrum) else dict(sp)
    for label, s in sps.items():
        if per_decade is not None:
            s = s.at_log_freqs(per_decade)
        ax.plot(s.freqs, s.power, label=label, **kwargs)
    if scale == "loglog":
        ax.set_xscale("log")
    if scale in ("loglog", "logy"):
        ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power spectral density (units$^2$/Hz)")
    ax.grid(True, which="major", alpha=0.3)
    if any(label is not None for label in sps):
        ax.legend(frameon=False)
    return ax
